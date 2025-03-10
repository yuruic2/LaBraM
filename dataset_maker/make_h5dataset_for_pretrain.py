from pathlib import Path
from shock.utils import h5Dataset
from shock.utils import preprocessing_cnt

savePath = Path('path/to/your/save/path')
rawDataPath = Path('path/to/your/raw/data/path')
group = rawDataPath.glob('*.mefd')

# preprocessing parameters
l_freq = 0.5
h_freq = 70.0
rsfreq = 200

# channel number * rsfreq
chunks = (62, rsfreq)

dataset = h5Dataset(savePath, 'dataset')
for mefdFile in group:
    print(f'processing {mefdFile.name}')
    eegData, chOrder = preprocessing_mefd(mefdFile, l_freq, h_freq, rsfreq)
    chOrder = [s.upper() for s in chOrder]
    eegData = eegData[:, :-10*rsfreq]
    grp = dataset.addGroup(grpName=cntFile.stem)
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # dataset attributes
    dataset.addAttributes(dset, 'lFreq', l_freq)
    dataset.addAttributes(dset, 'hFreq', h_freq)
    dataset.addAttributes(dset, 'rsFreq', rsfreq)
    dataset.addAttributes(dset, 'chOrder', chOrder)

dataset.save()
