import os
import re
import pdfplumber
from collections import Counter
import logging


def analyze_fonts_and_sizes(pdf):
    font_counter = Counter()
    size_counter = Counter()

    for page in pdf:
        for text in page.extract_text_lines():
            if text['text']:
                font = text['chars'][0]['fontname']
                size = round(text['chars'][0]['size'])
                font_counter[font] += 1
                size_counter[size] += 1

    fonts = font_counter
    sizes = size_counter
    return fonts, sizes


def checkRegEx(size, file):
    patternDotChapter = r'\d+\.\s.*$'
    patternFindChapter = r'.*chapter.*'
    for index, page in enumerate(file.pages):
        for text in page.extract_text_lines():
            if text['text']:
                if round(text['chars'][0]['size']) == size:
                    if re.findall(patternFindChapter, text['text'], flags=re.IGNORECASE) or re.findall(
                            patternDotChapter, text['text'], flags=re.IGNORECASE):
                        logging.info("Matched RegEx", text['text'])
                        return True
    return False


def checkStatistics(size, section):
    import numpy as np
    q1 = np.percentile(section, 25)
    q3 = np.percentile(section, 75)
    std = round(np.std(section), 1)
    #(size, "#sections =", len(section), "| var = ", round(np.var(section), 1), "| std ", std,
    #      "| coverage per section = ", round(100 * (q3 - q1 + np.mean(section)) / lastPage, 1), "| interQ = ", q3 - q1)
    return np.var(section), std, round(100 * (q3 - q1 + np.mean(section)) / lastPage, 1), q3 - q1


def create_size_dictionary(file, most_common_sizes):
    z = dict()
    for size in most_common_sizes:
        if most_common_sizes[size] > 50:
            continue
        if size <= textSize:
            continue
        x = []
        for index, page in enumerate(file.pages):
            for text in page.extract_text_lines():
                if text['text']:
                    if round(text['chars'][0]['size']) == size:
                        x.append(index)
        x = sorted(list(set(x)))
        if len(x) > 1:
            z[size] = x
    return z


def create_size_section_dictionary(size_dic):
    sections = dict()
    for size in size_dic.keys():
        startPage = 0
        coveragePages = []
        for cutPage in size_dic[size]:
            coveragePages.append(cutPage - startPage)
            startPage = cutPage
        coveragePages.append(lastPage - startPage)
        sections[size] = coveragePages[1:-1]
    return sections


def find_seperator_size(sections):
    validRegEx = []
    for size in sections.keys():
        if checkRegEx(size, file):
            validRegEx.append(size)

    if validRegEx:
        chapterSize = max(validRegEx)

    elif len(sections.keys()) == 1:
        chapterSize = sections.keys()[0]

    else:
        goodSizes = []
        coverageLimit = 20
        sectionLimit = 7
        while not goodSizes and coverageLimit <= 40:
            for size in sections.keys():
                section = sections[size]
                variance, std, coverage, interQ = checkStatistics(size, section)
                if not (len(section) < sectionLimit or variance < 1 or coverage > coverageLimit or interQ < 1):
                    goodSizes.append(size)
            if goodSizes:
                chapterSize = max(goodSizes)
            coverageLimit += 5
    return chapterSize

def is_majority_element(counter):
    if len(counter) == 0:
        return False

    most_common, most_common_count = counter.most_common(1)[0]
    total_count = sum(counter.values())

    return most_common_count > total_count * 0.5

def cut_pdf(pagesToCut, inputPDF):
    PDFs = []
    startPage = 0
    for index, cut in enumerate(pagesToCut):
        if startPage == cut:
            continue
        pdf_section = inputPDF.pages[startPage: cut]
        startPage = cut
        PDFs.append(pdf_section)
    PDFs.append(inputPDF.pages[startPage:])
    return PDFs

def clean_pdfs(pdfs, most_common_fonts):
    toDelete = []
    for part in pdfs:
        common_font, common_size = analyze_fonts_and_sizes(part)
        try:
            if not (common_font.most_common(1)[0][0] == most_common_fonts.most_common(1)[0][0] and is_majority_element(
                    common_size)):
                toDelete.append(part)
        except:
            toDelete.append(part)

    final_list = [x for x in pdfs if x not in toDelete]
    return final_list

def write_texts(sectionList, outputFolder):
    try:
        os.mkdir(outputFolder)
    except:
        logging.exception("Folder Already Exists or bad directory")
    for index, section in enumerate(sectionList):
        filename = outputFolder + str(index) + '.txt'
        with open(filename, 'w') as txt:
            for page in section:
                for line in page.extract_text_lines():
                    try:
                        if round(line['chars'][0]['size']) == textSize:
                            txt.write(line['text'] + '\n')
                    except:
                        continue

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    # Preprocess the texts
    preprocessed_texts = [text1, text2]

    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

    # Compute cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity
def findStartEnd(txt_files, folder):
    import os
    import numpy as np

    all= []
    for index1 in range(int(len(txt_files)/1.3)):
        com = []
        file1 = folder + str(index1)+'.txt'
        with open(file1,'r') as file:
            text1 = file.read()
        for index2 in range(int(len(txt_files)/1.3)):
            file2 = folder + str(index2)+'.txt'
            if index1 >= index2:
                continue
            with open(file2, 'r') as file:
                text2 = file.read()
            com.append(calculate_similarity(text1,text2))
        if com:
            all.append(com)
    s = []
    for a in all:
        s.append(np.mean(a))

    for index in range(len(s)):
        if np.mean(s)*0.9 < s[index]:
            start = index
            break

    all= []
    for index1 in range(start, len(txt_files)):
        com = []
        file1 = folder+ str(index1)+'.txt'
        with open(file1,'r') as file:
            text1 = file.read()
        for index2 in range(start,len(txt_files)):
            file2 = folder+ str(index2)+'.txt'
            if index1 <= index2:
                continue
            with open(file2, 'r') as file:
                text2 = file.read()
            com.append(calculate_similarity(text1,text2))
        if com:
            all.append(com)
    s = []
    for a in all:
        s.append(np.mean(a))
    for index in reversed(range(len(s))):
        if np.mean(s)*0.9 < s[index]:
            end = index + (len(txt_files)-len(s))
            break
    return start,end+1

def create_text_chunks(folder, starting_file, ending_file, chunk_length, chunk_overlap):
    chunk_folder = folder+'chunks'
    try:
        os.mkdir(chunk_folder)
    except FileExistsError:
        logging.exception("Folder Already Exists")
        pass
    for index in range(starting_file,ending_file+1):
        file_path = folder + str(index)+'.txt'

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        tokens = text.split() #nltk who?
        chunks = []
        start_index = 0

        while start_index < len(tokens):
            end_index = min(start_index + chunk_length, len(tokens))
            chunks.append(' '.join(tokens[start_index:end_index]))
            start_index += chunk_length - chunk_overlap

        for i, chunk in enumerate(chunks):
            filename = f"{chunk_folder}/{index}_chunk_{i}.txt"
            with open(filename, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)

if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    input = "input.pdf"
    file = pdfplumber.open(input)
    lastPage = len(file.pages) - 1
    most_common_fonts, most_common_sizes = analyze_fonts_and_sizes(file.pages)
    textSize = most_common_sizes.most_common(1)[0][0]
    size_dic = create_size_dictionary(file, most_common_sizes)
    sections = create_size_section_dictionary(size_dic)
    chapterSize = find_seperator_size(sections)
    pdfs = cut_pdf(size_dic[chapterSize], file)
    clean = clean_pdfs(pdfs, most_common_fonts)
    output_folder = 'output/'
    write_texts(clean, output_folder)
    print("happily Chunked!")
    txt_files = [file for file in os.listdir(output_folder) if file.endswith('.txt')]
    strt, end = findStartEnd(txt_files,output_folder)
    create_text_chunks(output_folder,strt, end, 2000, 250)
    print("happily wrote chunks to disks!")