from sacred import Experiment

ex = Experiment("ViLT")

def _loss_names(d):
    ret = {
        "bar": 0,
        "nico": 0,
        "imagenetLT": 0,
        "pacs": 0,
        'officehome': 0,
        'domain_net': 0,
    }
    ret.update(d)
    return ret

@ex.config
def config():
    exp_name = "vilt"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 4
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    # new features by beier
    load_path_prompt = ""
    validate_only = False
    finetune_mode = ""
    backbone_name = ""
    fix_backbone = False
    
    T = 1
    alpha = 0.3
    classname_path = ""
    WSL = False
    label_size = 6
    prompt_template = ''
    question_template = ''
    task = ''
    kd_loss = ''
    gamma = 1
    save_result = False
    test_domain = '' # hyperparamter for UDA 
    train_domain = ''


# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def task_finetune_bar():
    datasets = ["bar"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"bar": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-5
    val_check_interval = 0.5
    lr_mult = 1e2
    label_size = 6
    task = 'bar'


@ex.named_config
def task_finetune_nico():
    datasets = ["nico"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nico": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-5
    val_check_interval = 0.5
    lr_mult = 1e2
    label_size = 10
    task = 'nico'


@ex.named_config
def task_finetune_imagenetLT():
    datasets = ["imagenetLT"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"imagenetLT": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.5
    lr_mult = 10
    label_size = 1000
    task = 'imagenetLT'

@ex.named_config
def task_finetune_pacs():
    datasets = ["pacs"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"pacs": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-5
    val_check_interval = 0.5
    lr_mult = 1e2
    label_size = 10
    task = 'pacs'
    test_domain = 'sketch'


@ex.named_config
def task_finetune_domain_net():
    datasets = ["domain_net"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"domain_net": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-5
    val_check_interval = 0.5
    lr_mult = 1e2
    label_size = 345
    task = 'domain_net'
