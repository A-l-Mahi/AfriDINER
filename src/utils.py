import torch
import os
import random
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score,f1_score
from collections import defaultdict

def set_seed(manual_seed: int, n_gpu: int = 1):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(manual_seed)

def cal_ARS(ids,preds,golds):
    id_adic = {}
    cnt = 0
    for id,pred,gold in zip(ids,preds,golds):
        id = id.replace("_adv3","").replace("_adv2","").replace("_adv1","")
        if id not in id_adic:
            id_adic[id] = [[pred,gold]]
        else:
            id_adic[id].append([pred,gold])
    for i in id_adic:
        flag = 1
        for j in id_adic[i]:
            if j[0] != j[1]:
                flag = 0
        if flag:
            cnt += 1
    return cnt/len(id_adic)

def cal_acc_seperation(ids,preds,golds):
    base_gold = []
    base_pred = []
    adv1_pred = []
    adv1_gold = []
    adv2_pred = []
    adv2_gold = []
    adv3_pred = []
    adv3_gold = []
    for id,pred,gold in zip(ids,preds,golds):
        if "adv1" in id:
            adv1_pred.append(pred)
            adv1_gold.append(gold)
        elif "adv2" in id:
            adv2_pred.append(pred)
            adv2_gold.append(gold)
        elif "adv3" in id:
            adv3_pred.append(pred)
            adv3_gold.append(gold)
        else:
            base_pred.append(pred)
            base_gold.append(gold)
    base_acc = accuracy_score(base_gold,base_gold)
    adv1_acc = accuracy_score(adv1_gold,adv1_pred)
    adv2_acc = accuracy_score(adv2_gold,adv2_pred)
    adv3_acc = accuracy_score(adv3_gold,adv3_pred)
    return base_acc, adv1_acc, adv2_acc, adv3_acc

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,Counterfactual):
    model = model.train()
    losses = []
    predications = []
    golds = []
    scaler = torch.amp.GradScaler('cuda')
    for data in tqdm(data_loader):
        all_input_ids = data['all_input_ids'].to(device)
        all_attention_mask = data['all_attention_mask'].to(device)
        text_input_ids = data['text_input_ids'].to(device)
        text_attention_mask = data['text_attention_mask'].to(device)
        aspect_input_ids = data['aspect_input_ids'].to(device)
        aspect_attention_mask = data['aspect_attention_mask'].to(device)
        targets = data['polarities'].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            if Counterfactual:
                outputs, all_out, text_out, aspect_out = model(
                    all_input_ids,all_attention_mask, text_input_ids,text_attention_mask,aspect_input_ids, aspect_attention_mask,targets
                    )
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets) + loss_fn(text_out, targets) + loss_fn(aspect_out, targets)
            else:
                outputs = model(
                    text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask
                    )
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        predications.extend(preds.tolist())
        golds.extend(targets.tolist())
        losses.append(loss.item())
    return accuracy_score(golds,predications),f1_score(golds,predications,average="macro"),np.mean(losses)

def eval_model(model, data_loader, loss_fn, device,Counterfactual, epoch,save_dir,flag = 0):
    model = model.eval()
    losses = []
    predications = []
    golds = []
    ids = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            all_input_ids = data['all_input_ids'].to(device)
            all_attention_mask = data['all_attention_mask'].to(device)
            text_input_ids = data['text_input_ids'].to(device)
            text_attention_mask = data['text_attention_mask'].to(device)
            aspect_input_ids = data['aspect_input_ids'].to(device)
            aspect_attention_mask = data['aspect_attention_mask'].to(device)
            targets = data['polarities'].to(device)
            ids.extend(data["id"])
            if Counterfactual:
                outputs = model(
                    all_input_ids,all_attention_mask, text_input_ids,text_attention_mask,aspect_input_ids, aspect_attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, targets)
            else:
                outputs = model(
                    text_input_ids, text_attention_mask, aspect_input_ids, aspect_attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            predications.extend(preds.tolist()) 
            golds.extend(targets.tolist())
            losses.append(loss.item())
    ARS = cal_ARS(ids,predications,golds)
    if flag == 1:
        with open(save_dir+"/"+"predictions"+"epoch"+str(epoch)+".txt","w",encoding="utf-8") as f:
            for id,pred,gold in zip(ids,predications,golds):
                f.write(id+"\t"+str(pred)+"\t"+str(gold)+"\n")
    return accuracy_score(golds,predications),f1_score(golds,predications,average="macro"), ARS ,np.mean(losses)

def main(EPOCHS, MODEL, train_data_loader, val_data_loader, test_data_loader, loss_fn, optimizer, device, scheduler, save_dir, Counterfactual, patience=10):
    history = defaultdict(list)
    best_acc = 0
    best_f1 = 0
    best_ARS = 0
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(EPOCHS)):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_f1, train_loss = train_epoch(
            MODEL, train_data_loader, loss_fn, optimizer, device, scheduler, Counterfactual
        )
        print(f'Train loss {train_loss} acc {train_acc} f1 {train_f1}')

        val_acc, val_f1, val_ARS, val_loss = eval_model(
            MODEL,
            val_data_loader,
            loss_fn,
            device,
            Counterfactual,
            epoch,
            save_dir = save_dir
        )
        print(f'Val   loss {val_loss} acc {val_acc} f1 {val_f1}')
        test_acc, test_f1, test_ARS,test_loss = eval_model(
            MODEL,
            test_data_loader,
            loss_fn,
            device,
            Counterfactual,
            epoch,
            save_dir = save_dir,
            flag = 1
        )
        print(f'Test loss {test_loss} acc {test_acc} f1 {test_f1} ARS {test_ARS}')

        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_loss'].append(val_loss)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(test_f1)
        history['test_loss'].append(test_loss)
        history["ARS"].append(test_ARS)

        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_ARS = val_ARS
            epochs_no_improve = 0
            torch.save(MODEL.state_dict(), save_dir+'/best_model_state.bin')
        else:
            epochs_no_improve += 1

        print(f'Best acc {best_acc} best f1 {best_f1} best ARS {best_ARS}')

        with open(save_dir+"/result.txt", "w", encoding="utf-8") as f:
            f.write("best acc:"+str(max(history['test_acc'])))
            f.write("best f1:"+str(max(history['test_f1'])))
            f.write("best ARS:"+str(max(history['ARS'])))
            f.write("best epoch:"+str(epoch))

        if epochs_no_improve >= patience:
            print("Early stopping triggered. Stopping GCP VM...")
            break
    os.system("gcloud compute instances stop afri-semeval --discard-local-ssd=true --zone=us-central1-a")