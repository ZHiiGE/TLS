# TLS
The code of paper "Thinking Like Sonographers: Human-centeredCNN models for diagnosing gout from musculoskeletal ultrasound"

# Get base model
'python train_TLS.py --save_path ./model/base/ --use_cuda'

# Train without TLS
'python train_TLS.py --save_path ./model/TLS/ --load_path ./model/base/ --trainWithMap True --use_cuda '
