import torch
import torch.nn.functional as F


def generate_name(
    model,
    token2id,
    id2token,
    start_text="",          # начало имени (без SOS/EOS), например "m", "an"
    max_len=20,
    temperature=1.0,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model.eval()

    sos_token = "<"
    eos_token = ">"

    # 1. формируем начальную последовательность токенов
    if start_text:
        # без явного SOS в начале, если ты его не хочешь включать
        init_tokens = list(start_text.lower())
    else:
        # если пусто — начинаем строго с SOS
        init_tokens = [sos_token]

    # проверяем, что все символы в словаре
    for ch in init_tokens:
        if ch not in token2id:
            raise ValueError(f"Символ '{ch}' отсутствует в словаре токенов")

    # 2. переводим в индексы
    input_ids = [token2id[ch] for ch in init_tokens]
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)

    # начальное скрытое состояние
    hidden = model.init_hidden(batch_size=1, device=device)

    # 3. прогоняем начальную последовательность через модель,
    #    чтобы прогреть скрытое состояние
    with torch.no_grad():
        token_logits, _, _, hidden = model(input_tensor, hidden=hidden)

    # последнее сгенерированное имя (пока — просто старт)
    generated_tokens = init_tokens.copy()

    # 4. по шагам генерируем новые символы
    for _ in range(max_len):
        # берём последний токен как вход
        last_token_id = token2id[generated_tokens[-1]]
        last_input = torch.tensor([[last_token_id]], dtype=torch.long, device=device)  # (1, 1)

        with torch.no_grad():
            token_logits, _, _, hidden = model(last_input, hidden=hidden)
            # берём логиты последнего шага: (1, 1, vocab_size) -> (vocab_size,)
            logits = token_logits[:, -1, :].squeeze(0)

            # применяем temperature и softmax
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)  # (vocab_size,)

            # сэмплируем следующий токен по распределению
            next_id = torch.multinomial(probs, num_samples=1).item()
            next_token = id2token[next_id]

        # условие остановки
        if next_token == eos_token:
            break

        generated_tokens.append(next_token)

    # убираем служебные токены из финальной строки
    name_chars = [ch for ch in generated_tokens if ch not in (sos_token, eos_token)]
    name = "".join(name_chars)

    return name
