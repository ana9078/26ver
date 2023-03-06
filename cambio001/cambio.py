
import os
import torchaudio
#from speechbrain.dataio.dataio import read_audio

from speechbrain.pretrained import SepformerSeparation as separator
from direc02.direc002 import direccion



file_path= os.path.join(direccion,'audio-with-noise.wav')

model = separator.from_hparams(source="speechbrain/sepformer-wham", savedir='pretrained_models/sepformer-wham')

est_sources = model.separate_file(path=file_path)

torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)

torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)