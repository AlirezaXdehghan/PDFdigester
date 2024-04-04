import spacy
import nltk
import re
import fitz
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentIngestor:

    def __init__(self, file_path, max_chunk_length = 4800, min_chunk_length = 1000, header_height = 40, footer_height = 40):
        #self.text = self.extract_text_from_pdf(file_path)
        self.header_height = header_height
        self.footer_height = footer_height
        self.max_chunk_length = max_chunk_length
        self.min_chunk_length = min_chunk_length
        self._preprocessing_file(file_path)
        
    
    def _preprocessing_file(self, file_path):
        if file_path.endswith(".pdf"):
            self.text = self.extract_text_without_header_footer(file_path)
        elif file_path.endswith(".txt"):
            self.text = self.extract_text_from_txt(file_path)
        else:
            raise ValueError("Estensione del file non supportata")

    def extract_text_from_pdf(self, file_path):
        '''
        Extract text from pdf using fitz
        '''
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
          text+=page.get_text()

        text_preprocessed = self._preprocessing_text(text)

        return text_preprocessed
    
    def extract_text_from_txt(self, file_path):
        '''
        Extract text from txt
        '''
        with open(file_path, "r") as file:
            # Leggi il contenuto del file
            text = file.read()

        text_preprocessed = self._preprocessing_text(text)


        return text_preprocessed

    def extract_text_without_header_footer(self, pdf_path, start_page = 1):
        doc = fitz.open(pdf_path)
        all_text = ""
        end_page = doc.page_count

        for page_num in range(start_page - 1, end_page - 1 ):
            page = doc[page_num]

            rect = page.rect
            h_height = self.header_height
            f_height = self.footer_height
            rect.y1 -= h_height
            rect.y0 += f_height

            page_text = page.get_text("text", clip=rect)

            all_text += page_text

        doc.close()

        text_preprocessed = self._preprocessing_text(all_text)

        return text_preprocessed


    def get_NLTK_chunks(self):
        '''
        split text in chunks using the NLTK sentence tokenizer
        '''

        nltk.download('punkt')
        sentences_list = nltk.sent_tokenize(self.text)

        new_sentences_list = []
        for sent in sentences_list:
            if len(sent) > self.max_chunk_length:
                c_list = self._divide_chunk_text(sent)
                new_sentences_list.extend(c_list)
            else:
                new_sentences_list.append(sent)

        new_list = self._check_min_length(new_sentences_list)
        chunks_dictionary = self._create_dictionary(new_list)

        return chunks_dictionary

    def get_spacy_chunks(self):
        '''
        split text in chunks using the Spacy sentence tokenizer
        '''

        nlp = spacy.load('en_core_web_sm')
        nlp.max_length = len(self.text) + 100
        doc = nlp(self.text)
        sentences_list = [sent.text for sent in doc.sents]

        new_sentences_list = []
        for sent in doc.sents:
            if len(sent.text) > self.max_chunk_length:
                c_list = self._divide_chunk_text(sent.text)

                new_sentences_list.extend(c_list)
            else:
                new_sentences_list.append(sent.text)

        new_list = self._check_min_length(new_sentences_list)
        chunks_dictionary = self._create_dictionary(new_list)

        return chunks_dictionary


    def get_recursiveTextSplitter_chunks(self):
        '''
        split text in chunks using the Langchain's RecursiveCharacterTextSplitter
        '''

        custom_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap  = 20,
            length_function = len,
            separators = ["\n\n","\n",".", " ", ""]

        )
        sentences_list = custom_text_splitter.split_text(self.text)

        chunks_dictionary = self._create_dictionary(sentences_list)

        return chunks_dictionary

    def _divide_chunk_text(self, t):
      sentences = re.split(r'(?<=[.!?])\s+', t)
      chunks = []
      current_chunk = ""

      for sentence in sentences:
          if len(current_chunk) + len(sentence) < self.max_chunk_length:
              current_chunk = f"{current_chunk} {sentence}"
          else:
              chunks.append(current_chunk.strip())
              current_chunk = sentence

      if current_chunk:
          chunks.append(current_chunk.strip())

      final_chunks = []
      for chunk in chunks:
          if len(chunk) > self.max_chunk_length:
              words = chunk.split()
              current_chunk = ""
              for word in words:
                  if len(current_chunk) + len(word) < self.max_chunk_length:
                      current_chunk = f"{current_chunk} {word}"
                  else:
                      final_chunks.append(current_chunk.strip())
                      current_chunk = word
              if current_chunk:
                  final_chunks.append(current_chunk.strip())
          else:
              final_chunks.append(chunk)

      return final_chunks


    def _create_dictionary(self, sentences_list):
        '''
        Creates a dictionary from a list of strings(chunks)
        '''

        dictionary_list = []
        for i, sentence in enumerate(sentences_list):
                id = i
                length = len(sentence)
                dictionary = {"text": sentence, "id": str(i), "length": length}
                dictionary_list.append(dictionary)

        return dictionary_list

    def _check_min_length(self, chunks):
      new_list = []
      current_chunk = ""

      for chunk in chunks:
        if len(current_chunk) < self.min_chunk_length and len(current_chunk) + len(chunk) < self.max_chunk_length:
            current_chunk = f"{current_chunk} {chunk}"
        else:
            new_list.append(current_chunk)
            current_chunk = chunk

      new_list.append(current_chunk)

      return new_list


    def _preprocessing_text(self, original_text):
      pattern_point = r'([.!?])\1+'
      text_with_single_match = re.sub(pattern_point, r'\1', original_text)
      text_prepocessed = re.sub(r'[\xa0\t\r\n]', ' ', text_with_single_match)

      return text_prepocessed
