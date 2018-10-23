import pandas as pd
import os
import requests
import gzip


class getData(object):
    def __init__(self, link, output_dir='Desenvolvimento/Desafio/cleaning', filename=None):
        self.link = link
        self.output_dir = output_dir
        self.filename = filename
        
    def download_data(self):
        request = requests.get(self.link)
        content = request.content

        if self.filename is None:
            self.filename = self.link.split("/")[-1].replace("csv", "zip")
            
        with open(os.path.join(self.output_dir, self.filename), 'wb') as f:
            f.write(content)
            

    def decompress_file(self):
        filename = self.output_dir + "/" + self.filename
        with gzip.open(filename, 'rb') as f:
            content = f.read()

        decoded = content.decode('UTF-8')
        csv_fn = filename.replace('zip', 'csv')
        with open(csv_fn, 'w', encoding='utf-8') as f:
            f.write(decoded)

        print(csv_fn)
        
    def download(self):
        print('retrieving data ...')
        self.download_data()
        self.decompress_file()
        
        print("Done !")
        
if __name__ == '__main__':
    link = 'https://storage.googleapis.com/qedu-dados/escolas20070101000000000000.csv'
    
    gd = getData(link=link)
    gd.download()
    