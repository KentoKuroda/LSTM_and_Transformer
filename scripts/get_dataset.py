import os
import pickle
import json
import pandas as pd
import numpy as np
import io
import datetime as dt
import torch
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# Google Drive APIのスコープ
SCOPES = ['https://www.googleapis.com/auth/drive']


def init_dataset(sequence_np, label_np, batch_size, make_graph=False):
    # slice する数を計算
    slice_len = len(sequence_np) % 10
    sequence_np = np.delete(sequence_np, slice(len(sequence_np) - slice_len, len(sequence_np)), 0)
    label_np = np.delete(label_np, slice(len(sequence_np) - slice_len, len(sequence_np)), 0)

    # torch.tensorでtensor型に
    train_x = torch.from_numpy(sequence_np.astype(np.float32)).clone()
    train_t = torch.from_numpy(label_np.astype(np.float32)).clone()

    print('train_x:', train_x.shape)
    print('train_t:', train_t.shape)

    dataset = torch.utils.data.TensorDataset(train_x, train_t)

    if make_graph:
        testloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle = False, num_workers = 0)
        return testloader

    else:
        train_size = int(len(dataset) * 0.8) # train_size is 3000
        val_size = int(len(dataset) * 0.1) # val_size is 1000
        test_size = int(len(dataset) * 0.1)# val_size is 1000
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], torch.Generator().manual_seed(3)) # 42
        
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = True, num_workers = 0)
        valloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle = True, num_workers = 0)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = True, num_workers = 0)

        return trainloader, valloader, testloader


def googledrive_download(make_graph=False, bepro=False, classification=False):

    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    bepro_folder_ID = '15-uOWXpPpUSK3MHiO792ShCkF9OXQtD0'
    tactical_action_sequence_np_folder_ID = '1FidT_CGBPbs63cZkm9Ckl3ej8CO3syxG'
    tactical_action_label_np_folder_ID = '1jYLOh_V7czR1BlaOCMeWBKJsHIeRJigr'

    # フォルダ内のファイルをリスト化
    tactical_action_sequence_np_files = list_files_in_folder(service, tactical_action_sequence_np_folder_ID, kind_of_file = 'np')
    if not tactical_action_sequence_np_files:
        print('No sequence_half_np files found in folder.')
        return
    
    tactical_action_label_np_files = list_files_in_folder(service, tactical_action_label_np_folder_ID, kind_of_file = 'np')
    if not tactical_action_label_np_files:
        print('No label_half_np files found in folder.')
        return
    
    bepro_files = list_files_in_folder(service, bepro_folder_ID, kind_of_file = 'np')
    if not bepro_files:
        print('No bepro files found in folder.')
        return

    if make_graph:
        test_np_folder_ID = '1prcX_7Qd_EWZ-FkcYt-G9ELa9nv6qKIM'

        # フォルダ内のファイルをリスト化
        test_np_files = list_files_in_folder(service, bepro_folder_ID, kind_of_file = 'np') # test_np
        if not test_np_files:
            print('No sequence_half_np files found in folder.')
            return

        if bepro:
            data = '117093_09_22-10_07_'
            bepro_sequence_np_file_name = data + 'sequence_np.npy'
            bepro_label_np_file_name = data + 'label_np.npy'
            sequence_np_file = find_same_match_file(bepro_files, bepro_sequence_np_file_name)
            label_np_file = find_same_match_file(bepro_files, bepro_label_np_file_name)
        else:
            data='both_team_all_tactical_action_'
            tactical_action_sequence_np_file_name = data + 'sequence_np.npy' # transition_
            tactical_action_label_np_file_name = data + 'label_np.npy'
            sequence_np_file = find_same_match_file(tactical_action_sequence_np_files, tactical_action_sequence_np_file_name)
            label_np_file = find_same_match_file(tactical_action_label_np_files, tactical_action_label_np_file_name)

        # numpy 取得
        sequence_np = read_file(service, sequence_np_file['id'], kind_of_file = 'np')
        label_np = read_file(service, label_np_file['id'], kind_of_file = 'np')

        return sequence_np, label_np

    else:
        if bepro:
            if classification:
                bepro_sequence_np_file_name = 'sequence_np_including_future_classification.npy'
                bepro_label_np_file_name = 'label_np_including_future_classification.npy'
            else:
                bepro_sequence_np_file_name = 'sequence_np_including_future.npy'
                bepro_label_np_file_name = 'label_np_including_future.npy'
            sequence_np_file = find_same_match_file(bepro_files, bepro_sequence_np_file_name)
            label_np_file = find_same_match_file(bepro_files, bepro_label_np_file_name)
            sequence_np = read_file(service, sequence_np_file['id'], kind_of_file = 'np')
            label_np = read_file(service, label_np_file['id'], kind_of_file = 'np')
        else:
            data='both_team_all_tactical_action'
            tactical_action_sequence_np_file_name = data + '_sequence_np.npy' # transition_
            tactical_action_label_np_file_name = data + '_label_np.npy'
            sequence_np_file = find_same_match_file(tactical_action_sequence_np_files, tactical_action_sequence_np_file_name)
            label_np_file = find_same_match_file(tactical_action_label_np_files, tactical_action_label_np_file_name)
            # numpy 取得
            sequence_np = read_file(service, sequence_np_file['id'], kind_of_file = 'np')
            label_np = read_file(service, label_np_file['id'], kind_of_file = 'np')
            label_np = label_np[:, 1:]

        return sequence_np, label_np


def authenticate():
    creds = None
    # トークンファイルが存在する場合、それを読み込む
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # トークンが存在しないか、無効な場合、ログインして新しいトークンを取得する
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # トークンを保存して、将来のために保存する
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds


def list_files_in_folder(service, folder_id, kind_of_file):
    if kind_of_file == 'json':
        query = f"'{folder_id}' in parents and mimeType='application/json'"
    elif kind_of_file == 'csv':
        query = f"'{folder_id}' in parents and mimeType='text/csv'"
    elif kind_of_file == 'pkl':
        query = f"'{folder_id}' in parents and name contains '.pkl'"
    elif kind_of_file == 'folder':
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
    elif kind_of_file == 'np':
        query = f"'{folder_id}' in parents and (name contains '.npy' or name contains '.npz' or name contains 'np')"
    page_token = None
    items = []
    while True:
        response = service.files().list(q=query, fields="nextPageToken, files(id, name)", pageToken=page_token).execute()
        items.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return items


def find_same_match_file(files, file_name): # files の中から欲しい file_name を探して file を返す

    for file in files:
        if file['name'] == file_name:
            return_file = file
            break

    return return_file


def read_file(service, file_id, kind_of_file):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    fh.seek(0)
    if kind_of_file == 'json':
        return json.load(fh)
    elif kind_of_file == 'pkl':
        return pickle.load(fh)
    elif kind_of_file == 'csv':
        return pd.read_csv(fh, low_memory=False)
    elif kind_of_file == 'np':
        return np.load(fh, allow_pickle=True)