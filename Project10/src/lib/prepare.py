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
    
    return token2id, id2token