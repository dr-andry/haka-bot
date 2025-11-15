import re
import string
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pymorphy3

from dataclasses import dataclass
import logging
import json


# ==== Загрузка необходимых ресурсов NLTK ====
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# ============================================

# ========= Класс для обработки языка ========

class EventNLPProcessor:
    """
    Модуль NLP для обработки и анализа текстов мероприятий
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_resources()
        
    def setup_resources(self):
        """Инициализация языковых ресурсов"""
        # Русские стоп-слова
        self.russian_stopwords = set(stopwords.words('russian'))
        # Дополнительные стоп-слова для мероприятий
        self.custom_stopwords = {
            'мероприятие', 'событие', 'вебинар', 'лекция', 'семинар', 
            'конференция', 'время', 'место', 'регистрация', 'участие'
        }
        self.all_stopwords = self.russian_stopwords.union(self.custom_stopwords)
        
        # Стеммер и лемматизатор
        self.stemmer = SnowballStemmer('russian')
        self.morph_analyzer = pymorphy3.MorphAnalyzer()
        
        # Предопределенные категории мероприятий
        # Здесь все категории мероприятий в системе
        self.event_categories = {
            'programming': {'программирование', 'код', 'разработка', 'software', 'developer', 'backend', 'frontend'},
            'data_science': {'данные', 'анализ', 'ml', 'ai', 'машинное обучение', 'data science', 'нейросеть'},
            'design': {'дизайн', 'ui', 'ux', 'интерфейс', 'figma', 'adobe'},
            'business': {'бизнес', 'стартап', 'предпринимательство', 'маркетинг', 'продажи'},
            'career': {'карьера', 'трудоустройство', 'резюме', 'собеседование', 'hr'},
            'networking': {'нетворкинг', 'встреча', 'митап', 'сообщество'},
            'education': {'обучение', 'курс', 'образование', 'университет', 'студент'}
        }
        
        # Уровни сложности
        self.difficulty_levels = {
            'beginner': {'начальный', 'базовый', 'введение', 'для начинающих', 'starter', 'basic'},
            'intermediate': {'продолжающий', 'средний', 'углубленный', 'intermediate'},
            'advanced': {'продвинутый', 'advanced', 'профессиональный', 'экспертный', 'глубокий'}
        }

    def preprocess_text(self, text: str) -> str:
        """
        Предобработка текста: очистка, токенизация, лемматизация
        """
        if not text:
            return ""
            
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление пунктуации и цифр
        text = re.sub(f'[{string.punctuation}0-9]', ' ', text)
        
        # Токенизация
        tokens = word_tokenize(text, language='russian')
        
        # Лемматизация и фильтрация
        processed_tokens = []
        for token in tokens:
            if (len(token) > 2 and 
                token not in self.all_stopwords and 
                not token.isspace()):
                
                # Лемматизация с помощью pymorphy3
                parsed = self.morph_analyzer.parse(token)[0]
                lemma = parsed.normal_form
                processed_tokens.append(lemma)
        
        return ' '.join(processed_tokens)

    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Извлечение ключевых слов из текста с весами
        """
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return []
            
        # Используем TF-IDF для извлечения ключевых слов
        vectorizer = TfidfVectorizer(
            max_features=max_keywords * 2,
            ngram_range=(1, 2),  # Учитываем словосочетания
            min_df=1
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Сортируем ключевые слова по весу
            keywords_with_scores = [
                (feature_names[i], tfidf_scores[i]) 
                for i in range(len(feature_names)) 
                if tfidf_scores[i] > 0
            ]
            keywords_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keywords_with_scores[:max_keywords]
            
        except Exception as e:
            self.logger.warning(f"Ошибка при извлечении ключевых слов: {e}")
            return []

    def categorize_event(self, title: str, description: str) -> Dict[str, float]:
        """
        Классификация мероприятия по категориям
        Возвращает словарь с вероятностями принадлежности к категориям
        """
        full_text = f"{title} {description}"
        processed_text = self.preprocess_text(full_text)
        tokens = set(processed_text.split())
        
        category_scores = {}
        
        for category, keywords in self.event_categories.items():
            # Считаем пересечение токенов с ключевыми словами категории
            intersection = tokens.intersection(keywords)
            score = len(intersection) / len(keywords) if keywords else 0
            category_scores[category] = min(score * 10, 1.0)  # Нормализуем до [0, 1]
        
        return category_scores

    def detect_difficulty_level(self, title: str, description: str) -> str:
        """
        Определение уровня сложности мероприятия
        """
        full_text = f"{title} {description}".lower()
        
        level_scores = {}
        for level, keywords in self.difficulty_levels.items():
            score = sum(1 for keyword in keywords if keyword in full_text)
            level_scores[level] = score
        
        # Определяем доминирующий уровень
        if not any(level_scores.values()):
            return 'intermediate'  # Уровень по умолчанию
            
        return max(level_scores.items(), key=lambda x: x[1])[0]

    def calculate_relevance_score(self, event_text: str, 
                                specialty: str, 
                                interests: List[str]) -> float:
        """
        Расчет релевантности мероприятия для студента
        """
        processed_event = self.preprocess_text(event_text)
        processed_specialty = self.preprocess_text(specialty)
        processed_interests = [self.preprocess_text(interest) for interest in interests]
        
        # Объединяем все интересы и специальность
        query_text = f"{processed_specialty} {' '.join(processed_interests)}"
        
        # Векторизуем тексты
        vectorizer = CountVectorizer(ngram_range=(1, 2)).fit([query_text, processed_event])
        
        try:
            query_vector = vectorizer.transform([query_text])
            event_vector = vectorizer.transform([processed_event])
            
            # Косинусное сходство
            similarity = cosine_similarity(query_vector, event_vector)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.warning(f"Ошибка при расчете релевантности: {e}")
            return 0.0

    def extract_event_details(self, text: str) -> Dict:
        """
        Извлечение структурированной информации из текста мероприятия
        """
        details = {
            'duration': None,
            'format': None,
            'price': None,
            'language': None,
            'prerequisites': []
        }
        
        text_lower = text.lower()
        
        # Определение формата
        if any(word in text_lower for word in ['онлайн', 'online', 'zoom', 'webinar']):
            details['format'] = 'online'
        elif any(word in text_lower for word in ['офлайн', 'оффлайн', 'offline', 'очно']):
            details['format'] = 'offline'
        else:
            details['format'] = 'hybrid'
        
        # Определение стоимости
        if any(word in text_lower for word in ['бесплатно', 'free', 'бесплат']):
            details['price'] = 'free'
        elif any(word in text_lower for word in ['платно', 'paid', 'стоимость']):
            details['price'] = 'paid'
        else:
            details['price'] = 'unknown'
        
        # Определение языка
        if any(word in text_lower for word in ['английский', 'english', 'на английском']):
            details['language'] = 'english'
        else:
            details['language'] = 'russian'
        
        # Извлечение требований
        prerequisite_keywords = ['требования', 'необходимо', 'обязательно', 'prerequisites']
        for keyword in prerequisite_keywords:
            if keyword in text_lower:
                # Простая эвристика для извлечения требований
                sentences = re.split(r'[.!?]', text)
                for sentence in sentences:
                    if keyword in sentence.lower():
                        details['prerequisites'].append(sentence.strip())
        
        return details

    def recommend_by_course(self, events: List[Dict], course: int) -> List[Dict]:
        """
        Рекомендация мероприятий в зависимости от курса студента
        """
        recommended_events = []
        
        for event in events:
            difficulty = event.get('difficulty', 'intermediate')
            
            # Сопоставление курса и уровня сложности
            if course == 1:
                if difficulty in ['beginner', 'intermediate']:
                    recommended_events.append(event)
            elif course == 2:
                if difficulty in ['beginner', 'intermediate']:
                    recommended_events.append(event)
            elif course == 3:
                recommended_events.append(event)  # Все уровни
            else:  # 4 курс и магистратура
                if difficulty in ['intermediate', 'advanced']:
                    recommended_events.append(event)
        
        return recommended_events

    def analyze_event(self, event_data: Dict, specialty: str, interests: List[str]) -> Dict:
        """
        Полный анализ мероприятия
        """
        title = event_data.get('title', '')
        description = event_data.get('description', '')
        full_text = f"{title} {description}"
        
        # Извлекаем ключевые слова
        keywords = self.extract_keywords(full_text)
        
        # Классифицируем по категориям
        categories = self.categorize_event(title, description)
        
        # Определяем уровень сложности
        difficulty = self.detect_difficulty_level(title, description)
        
        # Рассчитываем релевантность
        relevance_score = self.calculate_relevance_score(full_text, specialty, interests)
        
        # Извлекаем детали
        details = self.extract_event_details(full_text)
        
        return {
            'original_event': event_data,
            'keywords': [kw[0] for kw in keywords[:5]],  # Топ-5 ключевых слов
            'categories': categories,
            'difficulty': difficulty,
            'relevance_score': relevance_score,
            'details': details,
            'processed_text': self.preprocess_text(full_text)
        }

    def rank_events(self, events: List[Dict], specialty: str, 
                   interests: List[str], course: int) -> List[Dict]:
        """
        Ранжирование мероприятий по релевантности
        """
        analyzed_events = []
        
        for event in events:
            analysis = self.analyze_event(event, specialty, interests)
            analyzed_events.append(analysis)
        
        # Сортируем по релевантности
        analyzed_events.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Фильтруем по курсу
        filtered_events = self.recommend_by_course(analyzed_events, course)
        
        return filtered_events

    def find_similar_events(self, target_event: Dict, events_pool: List[Dict], 
                           top_k: int = 5) -> List[Dict]:
        """
        Поиск похожих мероприятий на основе контента
        """
        target_text = self.preprocess_text(
            f"{target_event.get('title', '')} {target_event.get('description', '')}"
        )
        
        event_texts = [target_text]
        event_indices = [None]  # Индекс целевого события
        
        for i, event in enumerate(events_pool):
            event_text = self.preprocess_text(
                f"{event.get('title', '')} {event.get('description', '')}"
            )
            event_texts.append(event_text)
            event_indices.append(i)
        
        # Вычисляем косинусное сходство
        vectorizer = TfidfVectorizer().fit(event_texts)
        vectors = vectorizer.transform(event_texts)
        
        similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
        
        # Сортируем по схожести
        similar_indices = similarities.argsort()[::-1][:top_k]
        similar_events = [
            (events_pool[event_indices[i+1]], similarities[i]) 
            for i in similar_indices 
            if similarities[i] > 0.1  # Порог схожести
        ]
        
        return similar_events
# ============================================================



def rank(speciality: str, interests: list[str], course: int):

    nlp_processor = EventNLPProcessor()
    with open('./data/events_new.json', 'r', encoding='utf8') as f:
        events = json.load(f)
    
    ranked_events = nlp_processor.rank_events(events, speciality, interests, course)
    """
    ranked_events = [
            {
                'original_event': {
                        'title': str
                        'description': str
                        'url': str
                    }
                ,
                'keywords': [],  # Топ-5 ключевых слов мероприятия
                'categories': {  # численная оценка отношения мероприятия к данной категории
                    'programming': float, 
                    'data_science': float,
                    ....
                }, 
                'difficulty': string,
                'relevance_score': float,
                'details': {
                    'duration': None, 
                    'format': string, 
                    'price': string ot float, 
                    'language': string, 
                    'prerequisites': []
                },
                'processed_text': string
            }
        ]
    """

    
    return ranked_events

def get_top_k_for_person(person: dict, k: int):
    ranked_events = rank(
        person['speciality'],
        person['interests'],
        person['course']
    )

    min_treshold = 0.05

    output = []

    for event in ranked_events:
        if event['relevance_score'] > min_treshold:
            output.append(
                {
                    'title' : event['original_event']['title'],
                    'description': event['original_event']['description'],
                    'url': event['original_event']['url'],
                    'score': event['relevance_score']
                }
            )
        if len(output) == k:
            break

    print("DEBUG: get_top_k_for_person output:", output)

    return output

if __name__ == '__main__':
    # person = {
    #     'speciality': "компьютерные науки",
    #     'interests': ['машинное обучение', "нейронные сети", "математика", "поиск работы"],
    #     'course': 4
    # }
    k = 5
    person = {
            'speciality': "бизнес информатика",
            'interests': ['карьера', 'поиск работы', 'развитие', 'общение', 'бизнес', 'аналитика', 'экономика', 'ai-инструменты'],
            'course': 4
    }
    
    ranked_events = get_top_k_for_person(person=person, k=k)

    for (i, event) in enumerate(ranked_events):
        print(i)
        print(event['title'])
        print(event['description'])
        print(event['url'])

    