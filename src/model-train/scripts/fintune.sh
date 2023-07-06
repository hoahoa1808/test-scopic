mkdir -p ~/.huggingface
HUGGINGFACE_TOKEN = "*******" # token taken from Huggingface (free 100%)
echo -n "${HUGGINGFACE_TOKEN}" > ~/.huggingface/token

MODEL_NAME = "stable-diffusion-v1-5" 
OUTPUT_DIR = "./weights"            #forder to save model
mkdir -p $OUTPUT_DIR


# start running model
python3 train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \           # we used the pretrain model to avoid training from scratch
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \                                    
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=256 \                                                    #default is 512 but I had to reduce to 256 because of resource limitation
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \                                            # use this arg to speed up training
  --use_8bit_adam \                                                     # use this arg to save Ram memory
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \                                               
  --sample_batch_size=4 \
  --max_train_steps=8000 \                                              # Depend on the length of dataset
  --save_interval=50 \
  --save_sample_prompt="photo of student activity" \
  --concepts_list="concepts_list.json"                                  # `--save_sample_prompt` can be same as `--instance_prompt` to generate intermediate samples (saved along with weights in samples directory).


