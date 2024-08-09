pip install sentencepiece
pip install accelerate

python semeval/processing/xlmtocsv.py --dm str --lang str --tp str
python semeval/processing/datatoaspect.py --dm str --lang str --tp str
python semeval/processing/relabel.py --dm str --lang str --tp str --ck True

tensorboard --logdir=result/logs
