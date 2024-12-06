cd ~/cs229-music/clap-to-t5/

python infer_all.py --frozen=frozen --epoch=15 --dataset=train
python infer_all.py --frozen=frozen --epoch=15 --dataset=val
python infer_all.py --frozen=frozen --epoch=15 --dataset=test

python infer_all.py --frozen=unfrozen --epoch=15 --dataset=train
python infer_all.py --frozen=unfrozen --epoch=15 --dataset=val
python infer_all.py --frozen=unfrozen --epoch=15 --dataset=test

cd ~/cs229-music/wav2vec/

python infer_all.py --frozen=frozen --epoch=15 --dataset=train
python infer_all.py --frozen=frozen --epoch=15 --dataset=val

python infer_all.py --frozen=unfrozen --epoch=14 --dataset=train
python infer_all.py --frozen=unfrozen --epoch=14 --dataset=val

cd ~/cs229-music/mert_with_processor/

python infer_all.py --frozen=frozen --epoch=13 --dataset=train
python infer_all.py --frozen=frozen --epoch=13 --dataset=val

python infer_all.py --frozen=unfrozen --epoch=14 --dataset=train
python infer_all.py --frozen=unfrozen --epoch=14 --dataset=val