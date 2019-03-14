re-implemented speaker-listener-reinforcer (with ResNet)

## preprocess
```
python prepro.py -d refcocog -s google
```

## extract features
```
python scripts/extract_target_feats.py -d refcocog -s google --batch_size 40 -g 0
```

```
python scripts/extract_image_feats.py -d refcocog -s google --batch_size 40 -g 0
```

## training reinforcer
```
python scripts/train_vlsim.py -d refcocog -s google -g 0 --id slr
```

if you want to use attention in reinforcer and listener, please include 'attention' in --id.

## joint training (speaker, listerner with reinforcer's reward)
```
python train.py -d refcocog -s google -g 0 --id slr --id2 ver1
```