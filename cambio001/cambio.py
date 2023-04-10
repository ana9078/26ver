import os
import torchaudio
import torch
from fastapi import FastAPI,File,  Response,UploadFile

from fastapi.responses import FileResponse

from speechbrain.pretrained import SepformerSeparation as separator
from fastapi.responses import Response
import requests
import speechbrain as sb
import soundfile as sf
import tempfile
import shutil

app = FastAPI()

@app.post("/sound/")
async def create_upload_file(file: UploadFile = File(...)):


    model1 = separator.from_hparams(source="speechbrain/sepformer-wham", savedir='pretrained_models/sepformer-wham')
    model2 = separator.from_hparams(source="speechbrain/sepformer-libri2mix", savedir='pretrained_models/sepformer-libri2mix')
  
    audio_file_path=tempfile.NamedTemporaryFile(suffix=".wav",mode='w+b',delete=False)
    print(audio_file_path.name)

    with audio_file_path as buffer:
      shutil.copyfileobj(file.file, buffer)
      
    est_sources = model2.separate_file(audio_file_path.name)
    est_sources2 = model1.separate_file(audio_file_path.name)
    nome=tempfile.TemporaryDirectory()
    file_path01 = os.path.join(nome.name, "uno.wav")
    file_path02 = os.path.join(nome.name, "dos.wav")
    
    torchaudio.save(file_path01, est_sources[:, :, 0].detach().cpu(), 8000)
  
    torchaudio.save(file_path02, est_sources2[:, :, 0].detach().cpu(), 8000)
    filename = tempfile.NamedTemporaryFile()
    format = "zip"
    directory = nome.name
    shutil.make_archive(filename.name, format, directory)
    name_path=filename.name+'.zip'
    return FileResponse(name_path,media_type="application/octet-stream",filename="without_noice.zip")
   
   




   





    
