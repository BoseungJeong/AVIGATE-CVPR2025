import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_TrainDataLoader
from dataloaders.dataloader_vatex_retrieval import VATEX_DataLoader
from dataloaders.dataloader_charades_retrieval import Charades_DataLoader
def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path = args.audio_path,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.eval_max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path = args.audio_path,
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_vatex_train(args, tokenizer):
    vatex_dataset = VATEX_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path = args.audio_path,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(vatex_dataset)
    dataloader = DataLoader(
        vatex_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(vatex_dataset), train_sampler

def dataloader_vatex_test(args, tokenizer, subset="test"):
    vatex_testset = VATEX_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path = args.audio_path,
    )
    dataloader_vatex = DataLoader(
        vatex_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_vatex, len(vatex_testset)


def dataloader_charades_train(args, tokenizer):
    charades_dataset = Charades_DataLoader(
        csv_path=args.train_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path = args.audio_path,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(charades_dataset)
    dataloader = DataLoader(
        charades_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(charades_dataset), train_sampler

def dataloader_charades_test(args, tokenizer, subset="test"):
    charades_testset = Charades_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.eval_max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        audio_path = args.audio_path,
    )
    dataloader_charades = DataLoader(
        charades_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_charades, len(charades_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test, "test":None}
DATALOADER_DICT["vatex"] = {"train":dataloader_vatex_train, "val":dataloader_vatex_test, "test":dataloader_vatex_test}
DATALOADER_DICT["charades"] = {"train":dataloader_charades_train, "val":dataloader_charades_test, "test":dataloader_charades_test}
