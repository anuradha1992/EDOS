import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle
from os import mkdir
from os.path import exists, join
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

batch_size = 512
model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
emotions = ['afraid','angry','annoyed','anticipating','anxious','apprehensive','ashamed','caring',
    'confident','content','devastated','disappointed','disgusted','embarrassed','excited',
    'faithful','furious','grateful','guilty','hopeful','impressed','jealous','joyful',
    'lonely','nostalgic','prepared','proud','sad','sentimental','surprised','terrified',
    'trusting','agreeing','acknowledging','encouraging','consoling','sympathizing',
    'suggesting','questioning','wishing','neutral']

def get_os_sentence_embeddings(read_path, write_path):
    print('Reading the csv file...')
    df = pd.read_csv(read_path)
    N_rows = df.shape[0]

    uttrs = df.text.tolist()
    print('Creating the NumPy file...')
    sent_embed = np.empty((N_rows, 768), dtype = np.float32)
    N_batches = N_rows // batch_size
    print('Calculating the sentence embeddings...')
    for batch in tqdm(range(N_batches)):
        s = batch * batch_size
        t = s + batch_size
        sent_embed[s:t] = model.encode(uttrs[s:t])
    N_remain = N_rows % batch_size
    if N_remain != 0:
        sent_embed[-N_remain:] = model.encode(uttrs[-N_remain:])

    print('Saving the NumPy file to disk...')
    if not exists(write_path):
        mkdir(write_path)
    np.save(join(write_path, 'sentence_embeddings.npy'), sent_embed)

    print('Obtaining the dialog indices...')
    current_dialog_id = -1
    dialog_indices = {}
    s = 0
    for i in tqdm(range(N_rows)):
        dialog_id = df.iloc[i]['dialogue_id']
        if dialog_id != current_dialog_id:
            if current_dialog_id != -1:
                dialog_indices[current_dialog_id] = (s, i)
            s = i
            current_dialog_id = dialog_id
    dialog_indices[current_dialog_id] = (s, N_rows)

    print('Writing the dialog indices to disk...')
    sorted_dialog_indices = sorted([k for k in dialog_indices.keys()])
    with open(join(write_path, 'dialog_indices.csv'), 'w') as f:
        f.write('dialog_id,start,end\n')
        for k in sorted_dialog_indices:
            f.write('{},{},{}\n'.format(k, dialog_indices[k][0], dialog_indices[k][1]))

def get_amt_sentence_embeddings(read_path, write_path):
    print('Reading the csv file...')
    df = pd.read_csv(read_path)
    N_rows = df.shape[0]

    emot_dialog_ids = {}
    dialog_indices = {}
    sent_embed = []
    s = 0
    for i in tqdm(range(N_rows)):
        dialog_id = df.iloc[i]['DialogID']
        dialog = df.iloc[i]['Dialog']
        agreed_emot = df.iloc[i]['Agreed emotion (2/3 vote)']
        agreed_emot = agreed_emot.replace(' (Other)', '')
        if agreed_emot.lower() in emotions:
            if agreed_emot not in emot_dialog_ids:
                emot_dialog_ids[agreed_emot] = []
            emot_dialog_ids[agreed_emot].append(dialog_id)
            uttrs = dialog.split('\n')
            sent_embed.append(model.encode(uttrs))
            dialog_indices[dialog_id] = (s, s + len(uttrs))
            s += len(uttrs)

    print('Saving the NumPy file to disk...')
    sent_embed = np.concatenate(sent_embed)
    if not exists(write_path):
        mkdir(write_path)
    np.save(join(write_path, 'sentence_embeddings.npy'), sent_embed)

    print('Writing the dialog indices to disk...')
    sorted_dialog_indices = sorted([k for k in dialog_indices.keys()])
    with open(join(write_path, 'dialog_indices.csv'), 'w') as f:
        f.write('dialog_id,start,end\n')
        for k in sorted_dialog_indices:
            f.write('{},{},{}\n'.format(k, dialog_indices[k][0], dialog_indices[k][1]))

    print('Writing the emotion of dialog ids to disk...')
    with open(join(write_path, 'emot_dialog_ids.txt'), 'w') as f:
        for emot, dialog_ids in emot_dialog_ids.items():
            f.write('{},{}\n'.format(emot, ','.join([str(i) for i in dialog_ids])))

def get_dialog_embeddings(read_path):
    print('Reading the sentence embeddings...')
    sent_embed = np.load(join(read_path, 'sentence_embeddings.npy'))
    print('Reading the dialog indices...')
    dialog_indices = {}
    max_num_turns = -1
    sorted_dialog_indices = []
    with open(join(read_path, 'dialog_indices.csv'), 'r') as f:
        lines = f.read().splitlines()
        for line in tqdm(lines[1:]):
            dialog_id, s, t = line.split(',')
            dialog_id, s, t = int(dialog_id), int(s), int(t)
            sorted_dialog_indices.append(dialog_id)
            dialog_indices[dialog_id] = (s, t)
            if t - s > max_num_turns:
                max_num_turns = t - s
    weights = 2 ** np.arange(max_num_turns)

    print('Calculating the dialog embeddings...')
    dialog_embed = []
    for dialog_id in tqdm(sorted_dialog_indices):
        s, t = dialog_indices[dialog_id]
        num_turns = t - s
        embed = np.sum(sent_embed[s:t] * np.expand_dims(weights[:num_turns], axis = 1) / np.sum(weights[:num_turns]), axis = 0)
        dialog_embed.append(embed)
    dialog_embed = np.array(dialog_embed)

    print('Writing the dialog embeddings to file...')
    np.save(join(read_path, 'dialog_embeddings.npy'), dialog_embed)

def split_amt_dialogs(input_file, output_file_1, output_file_2, ratio):
    print('Reading the emotion of dialog indices...')
    output_1 = {}
    output_2 = {}
    with open('amt/{}'.format(input_file), 'r') as f:
        for line in f:
            items = line.strip().split(',')
            emot = items[0]
            dialog_ids = [int(i) for i in items[1:]]
            shuffle(dialog_ids)
            num_split = int(len(dialog_ids) * ratio)
            output_2[emot] = dialog_ids[:num_split]
            output_1[emot] = dialog_ids[num_split:]

    print('Writing the dialog ids to disk...')
    with open('amt/{}'.format(output_file_1), 'w') as f:
        for emot, dialog_ids in output_1.items():
            f.write('{},{}\n'.format(emot, ','.join([str(i) for i in dialog_ids])))
    with open('amt/{}'.format(output_file_2), 'w') as f:
        for emot, dialog_ids in output_2.items():
            f.write('{},{}\n'.format(emot, ','.join([str(i) for i in dialog_ids])))

def calculate_cosine_similarities():
    def read_files(read_path):
        print('Reading the dialog indices from {}...'.format(read_path))
        sorted_dialog_indices = []
        with open(join(read_path, 'dialog_indices.csv'), 'r') as f:
            lines = f.read().splitlines()
            for line in tqdm(lines[1:]):
                dialog_id, _, _ = line.split(',')
                sorted_dialog_indices.append(int(dialog_id))
        print('Reading the dialog embeddings from {}...'.format(read_path))
        dialog_embed = np.load(join(read_path, 'dialog_embeddings.npy'))
        return sorted_dialog_indices, np.float32(dialog_embed)

    os_sorted_dialog_indices, os_dialog_embed = read_files('os_all')
    amt_sorted_dialog_indices, amt_dialog_embed = read_files('amt')

    def read_dialog_ids(file_name):
        emot_dialog_ids = {}
        all_dialog_ids = []
        with open('amt/{}'.format(file_name), 'r') as f:
            for line in f:
                items = line.strip().split(',')
                emot = items[0]
                dialog_ids = [int(i) for i in items[1:]]
                all_dialog_ids += dialog_ids
                emot_dialog_ids[emot] = dialog_ids
        all_dialog_ids = sorted(all_dialog_ids)
        return emot_dialog_ids, all_dialog_ids

    print('Reading the train/val/test ids...')
    train_val_ids, all_train_val_ids = read_dialog_ids('train_val_ids.txt')
    test_ids, all_test_ids = read_dialog_ids('test_ids.txt')

    os_dialog_id_mapping = {dialog_id: i for i, dialog_id in enumerate(os_sorted_dialog_indices)}
    amt_dialog_id_mapping = {dialog_id: i for i, dialog_id in enumerate(amt_sorted_dialog_indices)}

    print('Constructing the row numbers...')
    all_train_val_ids_set = set(all_train_val_ids)
    os_eval_ids = [dialog_id for dialog_id in os_sorted_dialog_indices if dialog_id not in all_train_val_ids_set]
    os_rows = [os_dialog_id_mapping[dialog_id] for dialog_id in os_eval_ids]
    amt_rows = [amt_dialog_id_mapping[dialog_id] for dialog_id in all_train_val_ids]
    with open('os_eval_ids.txt', 'w') as f:
        f.write(','.join([str(i) for i in os_eval_ids]))
    with open('amt_eval_ids.txt', 'w') as f:
        f.write(','.join([str(i) for i in all_train_val_ids]))

    print('Calculating the cosine similarities...')
    kernel_matrix = cosine_similarity(amt_dialog_embed[amt_rows], os_dialog_embed[os_rows])

    print('Saving the similarity matrix to file...')
    np.save('cosine_similarity.npy', kernel_matrix)

def get_similar_dialogs(top_n_1, top_n_2):
    print('Reading the evaluated dialog ids...')
    with open('os_eval_ids.txt', 'r') as f:
        os_eval_ids = f.read().split(',')
    os_eval_ids = [int(i) for i in os_eval_ids]
    with open('amt_eval_ids.txt', 'r') as f:
        amt_eval_ids = f.read().split(',')
    amt_eval_ids = [int(i) for i in amt_eval_ids]
    amt_eval_id_mapping = {dialog_id: i for i, dialog_id in enumerate(amt_eval_ids)}

    def read_dialog_ids(file_name):
        emot_dialog_ids = {}
        with open('amt/{}'.format(file_name), 'r') as f:
            for line in f:
                items = line.strip().split(',')
                emot = items[0]
                dialog_ids = [int(i) for i in items[1:]]
                emot_dialog_ids[emot] = dialog_ids
        return emot_dialog_ids

    print('Reading the train/val/test ids...')
    train_val_ids = read_dialog_ids('train_val_ids.txt')
    test_ids = read_dialog_ids('test_ids.txt')

    print('Reading the cosine similarity matrix...')
    kernel_matrix = np.load('cosine_similarity.npy')

    print('For each emotion, get similar dialogs...')
    with open('similar_dialog_ids.csv', 'w') as f:
        f.write('emotion,amt_dialog_id,os_dialog_id,cos_sim\n')
        sorted_emots = sorted([emot for emot in train_val_ids.keys()])
        for emot in sorted_emots:
            print('Finding similar dialogs of emotion "{}"...'.format(emot))
            dialog_ids = train_val_ids[emot]
            amt_rows = [amt_eval_id_mapping[i] for i in dialog_ids]
            flattened = kernel_matrix[amt_rows].reshape(-1)
            print('Obtaining the top n of the flattened matrix...')
            ind = np.argpartition(flattened, -top_n_1)[-top_n_1:]
            ind = ind[np.argsort(flattened[ind])[::-1]]
            print('Writing to the csv file...')
            added_cols = set()
            for i in ind:
                row = i // kernel_matrix.shape[1]
                col = i % kernel_matrix.shape[1]
                if col not in added_cols:
                    added_cols.add(col)
                    f.write('{},{},{},{}\n'.format(emot, dialog_ids[row], os_eval_ids[col], flattened[i]))
                if len(added_cols) == top_n_2:
                    break
            print('Obtained {} similar dialogs.'.format(len(added_cols)))

def get_corresponding_dialogs(file_name):
    print('Reading the OS csv file...')
    df_os = pd.read_csv('os_2018_dialogs_emobert_reduce_freq.csv')
    print('Reading the AMT csv file...')
    df_amt = pd.read_csv('MTurkHitsAll_results.csv')

    print('Reading the similar dialog ids...')
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()
    lines = lines[1:]

    print('Counting the frequency of OS dialog ids...')
    dialog_id_freq = {}
    for i, line in tqdm(enumerate(lines), total = len(lines)):
        _, _, os_dialog_id, cos_sim = line.split(',')
        os_dialog_id = int(os_dialog_id)
        cos_sim = float(cos_sim)
        if os_dialog_id not in dialog_id_freq:
            dialog_id_freq[os_dialog_id] = []
        dialog_id_freq[os_dialog_id].append((cos_sim, i))

    print('Removing duplicate lines...')
    removed = set()
    for dialog_id in tqdm(dialog_id_freq.keys(), total = len(dialog_id_freq)):
        dialog_id_freq[dialog_id] = sorted(dialog_id_freq[dialog_id], key = lambda x: -x[0])
        for i in range(1, len(dialog_id_freq[dialog_id])):
            removed.add(dialog_id_freq[dialog_id][i][1])
    entries = []
    for i, line in tqdm(enumerate(lines), total = len(lines)):
        if i not in removed:
            emot, amt_dialog_id, os_dialog_id, cos_sim = line.split(',')
            entries.append((emot, int(amt_dialog_id), int(os_dialog_id), float(cos_sim)))

    print('Writing to output...')
    with open('similar_dialogs.csv', 'w') as f:
        f.write('emotion,cos_sim,amt_dialog_id,amt_dialog,os_dialog_id,os_dialog\n')
        for emot, amt_dialog_id, os_dialog_id, cos_sim in tqdm(entries):
            df_amt_ = df_amt[df_amt.DialogID == amt_dialog_id]
            amt_dialog = df_amt_.iloc[0]['Dialog'].split('\n')
            amt_dialog = ['- {}'.format(u) for u in amt_dialog]
            amt_dialog = '\n'.join(amt_dialog).replace('"', '""')
            df_os_ = df_os[df_os.dialogue_id == os_dialog_id]
            os_dialog = df_os_['text'].tolist()
            os_dialog = ['- {}'.format(u) for u in os_dialog]
            os_dialog = '\n'.join(os_dialog).replace('"', '""')
            f.write('{},{:.4f},{},"{}",{},"{}"\n'.format(emot, cos_sim, amt_dialog_id, amt_dialog, os_dialog_id, os_dialog))
    print('In total {} dialogs added.'.format(len(entries)))

def get_sample_dialogs():
    df_all = pd.read_csv('similar_dialogs.csv')
    N_rows = df_all.shape[0]
    emot_cnt = {}
    sample_rows = []
    for i in tqdm(range(N_rows)):
        emot = df_all.iloc[i]['emotion']
        if emot not in emot_cnt:
            emot_cnt[emot] = 0
        emot_cnt[emot] += 1
        if emot_cnt[emot] < 100:
            sample_rows.append(i)
    df_all.iloc[sample_rows].to_csv('similar_dialogs_sample.csv')

def filter_similar_dialogs(threshold = None):
    df = pd.read_csv('similar_dialogs.csv')
    df = df.drop_duplicates(subset = ['os_dialog'])
    df = df[df.amt_dialog != df.os_dialog]
    if threshold is not None:
        df = df[df.cos_sim >= threshold]
        print(df.shape)
        df.to_csv('similar_dialogs_{:03d}.csv'.format(int(threshold * 100)))
    else:
        print(df.shape)
        df.to_csv('similar_dialogs_all.csv')

def select_similar_dialogs():
    categories = ['Afraid', 'Angry', 'Caring', 'Confident', 'Content',
                  'Devastated', 'Disgusted', 'Faithful', 'Lonely',
                  'Prepared', 'Proud', 'Sad', 'Neutral']
    df = pd.read_csv('similar_dialogs_3k.csv')
    for c in categories:
        print(c, (df.emotion == c).sum())
    df_sel = df[df.emotion.isin(categories)]
    df_sel.to_csv('similar_dialogs_sel.csv')
    print(df_sel.shape)

def calculate_cosine_similarities_2():
    def read_files(read_path):
        print('Reading the dialog indices from {}...'.format(read_path))
        dialog_ids = []
        with open(join(read_path, 'dialog_indices.csv'), 'r') as f:
            lines = f.read().splitlines()
            for line in tqdm(lines[1:]):
                dialog_id, _, _ = line.split(',')
                dialog_ids.append(int(dialog_id))
        print('Reading the dialog embeddings from {}...'.format(read_path))
        dialog_embed = np.load(join(read_path, 'dialog_embeddings.npy'))
        return dialog_ids, np.float32(dialog_embed)

    os_dialog_ids, os_dialog_embed = read_files('os_all')
    os_dialog_id_mapping = {dialog_id: i for i, dialog_id in enumerate(os_dialog_ids)}

    def read_dialog_ids(file_name):
        all_dialog_ids = []
        with open('amt/{}'.format(file_name), 'r') as f:
            for line in f:
                items = line.strip().split(',')
                dialog_ids = [int(i) for i in items[1:]]
                all_dialog_ids += dialog_ids
        all_dialog_ids = sorted(all_dialog_ids)
        return all_dialog_ids

    print('Reading the amt dialog ids...')
    amt_dialog_ids = read_dialog_ids('emot_dialog_ids.txt')

    print('Reading the similar dialog ids...')
    df_sim = pd.read_csv('similar_dialogs_3k.csv')
    sim_dialog_ids = df_sim['os_dialog_id'].tolist()

    print('Reading the high confident dialog ids...')
    df_high = pd.read_csv('high_conf_data/high_conf_each_100.csv')
    high_dialog_ids = sorted(df_high['os_dialog_id'].tolist())

    os_eval_dialog_ids = set(os_dialog_ids) - set(amt_dialog_ids) - set(sim_dialog_ids) - set(high_dialog_ids)
    os_eval_dialog_ids = sorted(list(os_eval_dialog_ids))

    high_index = [os_dialog_id_mapping[i] for i in high_dialog_ids]
    os_eval_index = [os_dialog_id_mapping[i] for i in os_eval_dialog_ids]

    high_dialog_embed = os_dialog_embed[high_index]
    os_eval_dialog_embed = os_dialog_embed[os_eval_index]

    print('Calculating the cosine similarity...')
    kernel_matrix = cosine_similarity(high_dialog_embed, os_eval_dialog_embed)

    if not exists('embedding_2'):
        mkdir('embedding_2')

    print('Saving the similarity matrix to file...')
    np.save('embedding_2/cosine_similarity.npy', kernel_matrix)
    np.save('embedding_2/high_eval_ids.npy', np.array(high_dialog_ids))
    np.save('embedding_2/os_eval_ids.npy', np.array(os_eval_dialog_ids))

def get_similar_dialogs_2(top_n_1, top_n_2):
    print('Reading the evaluated dialog ids...')
    os_eval_ids = np.load('embedding_2/os_eval_ids.npy')
    high_eval_ids = np.load('embedding_2/high_eval_ids.npy')
    high_eval_id_mapping = {dialog_id: i for i, dialog_id in enumerate(high_eval_ids)}

    def read_emot_dialog_ids():
        df_high = pd.read_csv('high_conf_data/high_conf_each_100.csv')
        emot_dialog_ids = {}
        for i in range(df_high.shape[0]):
            emot = df_high.iloc[i]['emotion']
            dialog_id = df_high.iloc[i]['os_dialog_id']
            if emot not in emot_dialog_ids:
                emot_dialog_ids[emot] = []
            emot_dialog_ids[emot].append(dialog_id)
        return emot_dialog_ids

    print('Reading the emot dialog ids...')
    high_emot_dialog_ids = read_emot_dialog_ids()

    print('Reading the cosine similarity matrix...')
    kernel_matrix = np.load('embedding_2/cosine_similarity.npy')

    print('For each emotion, get similar dialogs...')
    with open('embedding_2/similar_dialog_ids.csv', 'w') as f:
        f.write('emotion,high_dialog_id,os_dialog_id,cos_sim\n')
        sorted_emots = sorted([emot for emot in high_emot_dialog_ids.keys()])
        for emot in sorted_emots:
            print('Finding similar dialogs of emotion "{}"...'.format(emot))
            dialog_ids = high_emot_dialog_ids[emot]
            high_rows = [high_eval_id_mapping[i] for i in dialog_ids]
            flattened = kernel_matrix[high_rows].reshape(-1)
            print('Obtaining the top n of the flattened matrix...')
            ind = np.argpartition(flattened, -top_n_1)[-top_n_1:]
            ind = ind[np.argsort(flattened[ind])[::-1]]
            print('Writing to the csv file...')
            added_cols = set()
            for i in ind:
                row = i // kernel_matrix.shape[1]
                col = i % kernel_matrix.shape[1]
                if col not in added_cols:
                    added_cols.add(col)
                    f.write('{},{},{},{}\n'.format(emot, dialog_ids[row], os_eval_ids[col], flattened[i]))
                if len(added_cols) == top_n_2:
                    break
            print('Obtained {} similar dialogs.'.format(len(added_cols)))

def get_corresponding_dialogs_2(file_name):
    print('Reading the OS csv file...')
    df_os = pd.read_csv('os_2018_dialogs_emobert_reduce_freq.csv')
    print('Reading the high conf csv file...')
    df_high = pd.read_csv('high_conf_data/high_conf_each_100.csv')

    print('Reading the similar dialog ids...')
    with open(file_name, 'r') as f:
        lines = f.read().splitlines()
    lines = lines[1:]

    print('Counting the frequency of OS dialog ids...')
    dialog_id_freq = {}
    for i, line in tqdm(enumerate(lines), total = len(lines)):
        _, _, os_dialog_id, cos_sim = line.split(',')
        os_dialog_id = int(os_dialog_id)
        cos_sim = float(cos_sim)
        if os_dialog_id not in dialog_id_freq:
            dialog_id_freq[os_dialog_id] = []
        dialog_id_freq[os_dialog_id].append((cos_sim, i))

    print('Removing duplicate lines...')
    removed = set()
    for dialog_id in tqdm(dialog_id_freq.keys(), total = len(dialog_id_freq)):
        dialog_id_freq[dialog_id] = sorted(dialog_id_freq[dialog_id], key = lambda x: -x[0])
        for i in range(1, len(dialog_id_freq[dialog_id])):
            removed.add(dialog_id_freq[dialog_id][i][1])
    entries = []
    for i, line in tqdm(enumerate(lines), total = len(lines)):
        if i not in removed:
            emot, high_dialog_id, os_dialog_id, cos_sim = line.split(',')
            entries.append((emot, int(high_dialog_id), int(os_dialog_id), float(cos_sim)))

    print('Writing to output...')
    with open('embedding_2/similar_dialogs.csv', 'w') as f:
        f.write('emotion,cos_sim,high_dialog_id,high_dialog,os_dialog_id,os_dialog\n')
        for emot, high_dialog_id, os_dialog_id, cos_sim in tqdm(entries):
            df_high_ = df_high[df_high['os_dialog_id'] == high_dialog_id]
            high_dialog = df_high_.iloc[0]['os_dialog'].split('\n')
            high_dialog = ['- {}'.format(u) for u in high_dialog]
            high_dialog = '\n'.join(high_dialog).replace('"', '""')
            df_os_ = df_os[df_os.dialogue_id == os_dialog_id]
            os_dialog = df_os_['text'].tolist()
            os_dialog = ['- {}'.format(u) for u in os_dialog]
            os_dialog = '\n'.join(os_dialog).replace('"', '""')
            f.write('{},{:.4f},{},"{}",{},"{}"\n'.format(emot, cos_sim, high_dialog_id, high_dialog, os_dialog_id, os_dialog))
    print('In total {} dialogs added.'.format(len(entries)))

def filter_similar_dialogs_2(threshold = None):
    df = pd.read_csv('embedding_2/similar_dialogs.csv')
    df = df.drop_duplicates(subset = ['os_dialog'])
    df = df[df.high_dialog != df.os_dialog]
    if threshold is not None:
        df = df[df.cos_sim >= threshold]
        print(df.shape)
        df.to_csv('embedding_2/similar_dialogs_{:03d}.csv'.format(int(threshold * 100)))
    else:
        print(df.shape)
        df.to_csv('embedding_2/similar_dialogs_all.csv')

if __name__ == '__main__':
    # get_os_sentence_embeddings('os_2018_dialogs_emobert_reduce_freq.csv', 'os_all')
    # get_amt_sentence_embeddings('MTurkHitsAll_results.csv', 'amt')
    # get_dialog_embeddings('os_all')
    # get_dialog_embeddings('amt')
    # split_amt_dialogs()
    # split_amt_dialogs('train_val_ids.txt', 'train_ids.txt', 'val_ids.txt', 0.25)
    # calculate_cosine_similarities()
    # get_similar_dialogs(10000, 1000)
    # get_corresponding_dialogs('similar_dialog_ids.csv')
    # get_sample_dialogs()
    # filter_similar_dialogs()
    # select_similar_dialogs()
    # calculate_cosine_similarities_2()
    # get_similar_dialogs_2(10000, 1000)
    # get_corresponding_dialogs_2('embedding_2/similar_dialog_ids.csv')
    filter_similar_dialogs_2(0.9)
