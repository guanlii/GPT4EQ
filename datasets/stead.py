from .base import DatasetBase
from typing import Optional,Tuple
import os
import pandas as pd
import numpy as np
from operator import itemgetter
import h5py
from utils import logger
from ._factory import register_dataset
import re


class STEAD(DatasetBase):

    _name = "stead"
    _part_range = None  # (inclusive,exclusive)
    _channels = ["e", "n", "z"]
    _sampling_rate = 100

    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
        few_shot_ratio: float = 1,
        **kwargs
    ):
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
            few_shot_ratio=few_shot_ratio,
        )
        print(f"Few-shot ratio received: {self._few_shot_ratio}")

    def _load_meta_data(self,filename="chunk2.csv") -> pd.DataFrame:
        
        meta_df = pd.read_csv(
            os.path.join(self._data_dir, filename),
            low_memory=False,
        )

        for k in meta_df.columns:
            if meta_df[k].dtype in [object, np.object_, "object", "O"]:
                meta_df[k] = meta_df[k].str.replace(" ", "")

        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

        meta_df.reset_index(drop=True, inplace=True)

        if self._data_split:
            irange = {}
            irange["train"] = [0, int(self._train_size * meta_df.shape[0]*self._few_shot_ratio)]
            irange["val"] = [
                int(self._train_size * meta_df.shape[0]),
                int(self._train_size * meta_df.shape[0]) + int(self._val_size * meta_df.shape[0]),
            ]
            irange["test"] = [irange["val"][1], meta_df.shape[0]]

            r = irange[self._mode]
            meta_df = meta_df.iloc[r[0] : r[1], :]


            logger.info(f"Data Split: {self._mode}: {r[0]}-{r[1]}")

        return meta_df

    def _load_event_data(self, idx:int) -> Tuple[dict,dict]:
        """Load evnet data

        Args:
            idx (int): Index.

        Raises:
            ValueError: Unknown 'mag_type'

        Returns:
            dict: Data of event.
            dict: Meta data.
        """        
    
        target_event = self._meta_data.iloc[idx]
        
        trace_name = target_event["trace_name"]


        path = os.path.join(self._data_dir, f"chunk2.hdf5")
        with h5py.File(path, "r") as f:
            data = f.get(f"data/{trace_name}")
            data = np.array(data).astype(np.float32).T
            data = np.nan_to_num(data)

        (
            ppk,
            spk,
            mag_type,
            evmag,
            snr_str,
        ) = itemgetter(
            "p_arrival_sample",
            "s_arrival_sample",
            "source_magnitude_type",
            "source_magnitude",
            "snr_db",
        )(
            target_event
        )

        #assert mag_type.lower()=="ml"

        evmag = np.clip(evmag, 0, 8, dtype=np.float32)
        snr_str = snr_str.strip('[]')  # 去除字符串两端的方括号
        snrs = re.findall(r"\d+\.\d+|\d+", snr_str)  # 使用正则表达式提取数字
        snrs = [float(s) if s != "nan" else 0. for s in snrs]  # 将非"nan"的字符串转换为浮点数，将"nan"转换为0
        snr = np.array(snrs)  # 将列表转换为NumPy数组

        event = {
            "data": data,
            "ppks": [int(ppk)] if pd.notnull(ppk) else [],
            "spks": [int(spk)] if pd.notnull(spk) else [],
            "emg": [evmag] if pd.notnull(evmag) else [],
            "clr": [0] , # For compatibility with other datasets
            "snr": snr,
        }

        return event,target_event.to_dict()


class STEAD_light(STEAD):
    _name = "stead_light"
    _part_range = None
    _channels = ["e", "n", "z"]
    _sampling_rate = 100

    def __init__(
        self,
        seed: int,
        mode: str,
        data_dir: str,
        shuffle: bool = True,
        data_split: bool = True,
        train_size: float = 0.8,
        val_size: float = 0.1,
        **kwargs
    ):
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
        )

    def _load_meta_data(self,filename = f"chunk2_light.csv") -> pd.DataFrame:
        return super()._load_meta_data(filename=filename)

    def _load_event_data(self, idx: int) -> Tuple[dict,dict]:
        """Load event data

        Args:
            idx (int): Index of target row.

        Returns:
            dict: Data of event.
            dict: Meta data.
        """        
        return super()._load_event_data(idx=idx)



@register_dataset
def stead(**kwargs):
    dataset = STEAD(**kwargs)
    return dataset


@register_dataset
def stead_light(**kwargs):
    dataset = STEAD_light(**kwargs)
    return dataset