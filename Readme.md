# h2p2 - Face Verification 

# TO:Do
- Evaluate to automatically run on device(cpu or gpu)
- functional with gui to load two images and verify it
## Requirements
```
    virtualenv -p python3 .env
    source .env/bin/activate
    pip install -r requirements.txt
```
Diactivate venv
``` 
decactivate 
```


### Aws Ec2 venv Setup 

```
source activate pytorch_p36 
```

Train model

``` 
python train.py --data_dir data/face_verf --model_dir expirements/basemodel
```

Evaluate model

```
python evaluate.py --data_dir data/face_verf --model_dir expirements/basemodel --restore_file last
```

# Template 
https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision
