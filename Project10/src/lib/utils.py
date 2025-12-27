from sklearn.metrics import roc_auc_score
import torch

def build_vocab(names_processed):
    """
    names_processed: итерируемый объект строк
        (например, df["name_processed"])
    
    Возвращает:
        tokens      - отсортированный список уникальных символов
        token2id    - dict: символ -> индекс
        id2token    - dict: индекс -> символ
    """
    # 1. собираем множество всех символов из всех имён
    all_chars = set()
    for name in names_processed:
        all_chars.update(name)
    # 2. делаем детерминированный порядок (например, сортировка)
    tokens = sorted(all_chars)
    
    # 3. строим отображения
    token2id = {ch: i for i, ch in enumerate(tokens)}
    id2token = {i: ch for ch, i in token2id.items()}
    
    return tokens, token2id, id2token


def compute_roc_auc_splits(model, train_loader, valid_loader, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()

    def collect_probs_and_targets(loader):
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device).float().view(-1)

                # используем predict_gender
                probs, _ = model.predict_gender(x_batch, device=device)  # (batch, 1)
                probs = probs.view(-1)  # (batch,)

                all_probs.append(probs.cpu())
                all_targets.append(y_batch.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_targets = torch.cat(all_targets).numpy()
        return all_targets, all_probs

    y_train, p_train = collect_probs_and_targets(train_loader)
    y_valid, p_valid = collect_probs_and_targets(valid_loader)
    y_test,  p_test  = collect_probs_and_targets(test_loader)

    roc_train = roc_auc_score(y_train, p_train)
    roc_valid = roc_auc_score(y_valid, p_valid)
    roc_test  = roc_auc_score(y_test,  p_test)

    print(
    f"train_roc_auc: {roc_train:.3f}\n"
    f"valid_roc_auc: {roc_valid:.3f}\n"
    f"test_roc_auc:  {roc_test:.3f}"
)