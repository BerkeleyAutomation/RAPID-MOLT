'''
This script downloads all files (images) from the Google Drive and saves them locally at a specified location.
'''

from __future__ import print_function
import io
import os
from apiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/drive'

# Settings of experiment (the name of the folder) and the directory to save the images in
experiment = '2019_exp_10/'
images_dir = os.path.dirname(os.path.abspath(__file__)) + experiment

def main():
    prog_dir = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(images_dir)
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    num_of_files = 100
    while num_of_files > 0:
        store = file.Storage(prog_dir + '/token.json')
        creds = store.get()
        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets(prog_dir + '/credentials.json', SCOPES)
            creds = tools.run_flow(flow, store)
        service = build('drive', 'v3', http=creds.authorize(Http()))

        # Call the Drive v3 API
        param = {}
        results = service.files().list(**param).execute()
        items = results.get('files', [])
        i = 1
        num_of_files = len(items)
        print(len(items))
        for item in items:
            # print(item['name'])
            request = service.files().get_media(fileId=item['id'])
            fh = io.FileIO(images_dir + item['name'], 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            if done:
                service.files().delete(fileId=item['id']).execute()
                print(i, item['name'])
                i = i + 1


if __name__ == '__main__':
    main()
