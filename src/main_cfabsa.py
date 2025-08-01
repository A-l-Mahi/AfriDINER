from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch import nn
from data_utills import *
from model import ABSAmodel,CFABSAmodel, CFABSA_XLMR
from utils import *
import argparse


parser = argparse.ArgumentParser(description='CFABSA finetuning')
parser.add_argument("--ARTS",type=int)
parser.add_argument("--Counterfactual",type=int)
parser.add_argument("--GPU",type=str)
parser.add_argument("--fusion_mode",type=str)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--seed', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--max_len_s', type=int)
parser.add_argument('--max_len_a', type=int)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--test_model', type=int)
parser.add_argument('--checkpoint', type=int)
parser.add_argument('--test_model_path', type=str)
args = parser.parse_args()

args.ARTS = True if args.ARTS == 1 else False

if args.ARTS:
    print("Conducting inference on ARTS")
else:
    print(f"fintuning {args.model_name} on {args.dataset_name} dataset")

if args.Counterfactual:
    print("mode: counterfactual")
else:
    print("mode: original")

if args.model_name:
    if args.model_name == "XLMr":
        pretrained = AutoModel.from_pretrained("FacebookAI/xlm-roberta-large")
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

    elif args.model_name == "Afro-XLMr": 
        pretrained = AutoModel.from_pretrained("Davlan/afro-xlmr-large")
        tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-large")
    elif args.model_name == "Afro-XLMr-large":
        pretrained = AutoModel.from_pretrained("Davlan/afro-xlmr-large-76L")
        tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-large-76L")
    else:
        pretrained = AutoModel.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


    MODEL = CFABSA_XLMR(pretrained) if args.Counterfactual else ABSAmodel(pretrained)
    

    mode = "ARTS" if args.ARTS else "ORI"

    train_data_loader = {}
    val_data_loader = {}
    test_data_loader = {}
    adv1_data_loader = {}
    adv2_data_loader = {}
    adv3_data_loader = {}


    if args.ARTS:
        # Load the ARTS dataset
        adv1, adv2, adv3 = load_data(args.dataset_name, mode)

        for key, values in adv1.items():
            adv1_data_loader[key] = create_data_loader(values, tokenizer, max_len = args.max_len_s, max_len_a= args.max_len_a, batch_size=args.batch_size)
        for key, values in adv2.items():
            adv2_data_loader[key] = create_data_loader(values, tokenizer, max_len = args.max_len_s, max_len_a= args.max_len_a, batch_size=args.batch_size)
        for key, values in adv3.items():
            adv3_data_loader[key] = create_data_loader(values, tokenizer, max_len = args.max_len_s, max_len_a= args.max_len_a, batch_size=args.batch_size)

    else:
        # Load the original dataset
        train, test, dev = load_data(args.dataset_name, mode)
        train_data_loader = create_data_loader(train, tokenizer, max_len = args.max_len_s, max_len_a= args.max_len_a,batch_size=args.batch_size)

        val_data_loader = create_data_loader(dev, tokenizer, max_len = args.max_len_s, max_len_a= args.max_len_a, batch_size = args.batch_size)
        test_data_loader = {}

        for key, values in test.items():
            test_data_loader[key] = create_data_loader(values, tokenizer, max_len = args.max_len_s, max_len_a= args.max_len_a,  batch_size = args.batch_size)


device = torch.device("cuda:"+args.GPU if torch.cuda.is_available() else "cpu")
set_seed(args.seed)
my_model = MODEL.to(device)

total_steps = len(train_data_loader) * args.epoch

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in my_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in my_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=int(total_steps*0.2),
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)
parser = argparse.ArgumentParser(description='CFABSA finetuning')
save_dir = args.save_dir

if args.test_model and args.checkpoint:
    model_state = torch.load(args.test_model_path)

    MODEL.load_state_dict(model_state)

    my_model = MODEL.to(device)

    if args.ARTS:
        print("Testing on ARTS dataset")
        test_model(my_model, adv1_data_loader, loss_fn, device, args.Counterfactual, model_name=args.model_name, save_dir = "output/test_results/adv1", mode="test")
        test_model(my_model, adv2_data_loader, loss_fn, device, args.Counterfactual, model_name=args.model_name, save_dir = "output/test_results/adv2", mode="test")
        test_model(my_model, adv3_data_loader, loss_fn, device, args.Counterfactual, model_name=args.model_name, save_dir = "output/test_results/adv3", mode="test")
    else:
        print("Testing on original dataset")
        test_model(my_model, values, loss_fn, device, args.Counterfactual, model_name=args.model_name, save_dir = "output/test_results", mode="test")
    
else:
    main(args.epoch, my_model, train_data_loader, val_data_loader, test_data_loader, loss_fn,
        optimizer, device, scheduler, save_dir, args.Counterfactual, args.model_name)
