import os
from torch.utils.data import Dataset
import glob
import torchaudio
from murenn.dtcwt.utils import fix_length


CLASS_MAP = {
    'fan': 0,
    'pump': 1,
    'slider': 2,
    'valve': 3,
}
INVERSE_CLASS_MAP = {
    0: 'fan',
    1: 'pump',
    2: 'slider',
    3: 'valve',
}
SNR_MAP = {
    '0dB': 0,
    '6dB': 6,
    'min6dB': -6,
}
INVERSE_SNR_MAP = {
    0: '0dB',
    6: '6dB',
    -6: 'min6dB',
}




def enumerate_development_datasets():
    typ_id = []
    for snr in [0, 6, -6]:
        for i in range(4):
            for j in [0, 2, 4, 6]:
                for normal in [True, False]:
                    typ_id.append((snr, i, j, normal))
    return typ_id


class BaseDataSet(Dataset):
    def __init__(
            self,
            snr=0,
            machine_type=0,
            machine_id=0,
            normal=True,
            label=0,
            normalize_raw=True,
            data_path=None,
    ):
        super(BaseDataSet, self).__init__()

        self.data_path = data_path
        self.snr = snr
        self.machine_type = machine_type
        self.machine_id = machine_id
        self.type = 'normal' if normal else 'abnormal'
        self.label = label
        self.normalize_raw = normalize_raw
        self.data_list = self.__get_file_list__()


    def __getitem__(self, idx):
        audio_path = self.data_list[idx]
        # the audio was recorede with 8 microphone, we take the first channel
        sample,_ = torchaudio.load(audio_path)
        sample = fix_length(sample[0, :], size=160000)
        if self.normalize_raw:
            sample = sample / (sample.std() + 1e-8)
        return {
            'sample': sample,
            'label': float(self.label),
            'snr': self.snr,
            'machine_type': self.machine_type,
            'machine_id': self.machine_id,
            'file_id': os.path.basename(audio_path),
        }

    def __len__(self):
        return len(self.data_list)

    def __get_file_list__(self):
        files = glob.glob(
            os.path.join(
                self.data_path, INVERSE_SNR_MAP[self.snr], INVERSE_CLASS_MAP[self.machine_type],
                f'*id_{self.machine_id:02d}*',  self.type, '*.wav'
            )
        )
        assert len(files) > 0
        return sorted(files)



if __name__ == '__main__':
    for snr_, type_, id_, normal_ in enumerate_development_datasets():
        _ = BaseDataSet(snr_, type_, id_, normal_, data_path="/Users/zhang/MuReNN/data/MIMII")
        print(len(_))
    print(_[0])
