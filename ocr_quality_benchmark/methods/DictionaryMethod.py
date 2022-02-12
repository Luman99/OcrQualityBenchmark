from ocr_quality_benchmark.methods.Constants import alphabet


def dictionary_method(name: str, path_to_json: str) -> float:
    text2 = ''
    with open("D:\Inżynierka\Projekty\data\dev-0\in.tsv", encoding='latin1') as file:
        for line in file:
            if name in line:
                text2 = (line[53:])

    correct_words = 0
    all_words = 0
    with open(alphabet) as alp:
        for word in text2.split(' '):
            all_words += 1
            if word in alp:
                correct_words += 1

    return float(correct_words/all_words)
