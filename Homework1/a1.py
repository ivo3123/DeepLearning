#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################

### Домашно задание 1
###
### За да работи програмата трябва да се свали корпус от публицистични текстове за Югоизточна Европа,
### предоставен за некомерсиално ползване от Института за български език - БАН
###
### Корпусът може да бъде свален от:
### Отидете на http://dcl.bas.bg/BulNC-registration/#feeds/page/2
### И Изберете:
###
### Корпус с новини
### Корпус от публицистични текстове за Югоизточна Европа.
### 27.07.2012 Български
###	35337  7.9M
###
### Архивът трябва да се разархивира в директорията, в която е програмата.
###
### Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
### conda activate tii
###
### Ако все още нямате създадено обкръжение прочетете файла README.txt за инструкции

import langmodel
import math
import numpy as np
from tqdm import tqdm

def editDistance(s1: str, s2: str) -> np.ndarray:
	#### функцията намира разстоянието на Левенщайн - Дамерау между два низа
	#### вход: низовете s1 и s2
	#### изход: матрицата M с разстоянията между префиксите на s1 и s2 (виж по-долу)

    M = np.zeros((len(s1)+1,len(s2)+1))

	#### M[i,j] следва да съдържа разстоянието между префиксите s1[:i] и s2[:j]
	#### M[len(s1),len(s2)] следва да съдържа разстоянието между низовете s1 и s2
	#### За справка разгледайте алгоритъма editDistance от слайдовете на Лекция 1
	
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 15-30 реда
    
    d_t_lengths_levenshtein_damerau_distance = {}

    def _levenshtein_damerau_distance(s1: str, s2: str) -> int:
        cached_res = d_t_lengths_levenshtein_damerau_distance.get((len(s1), len(s2)))

        if cached_res is not None:
            return cached_res
        
        res = min(
            0 if len(s1) == 0 and len(s2) == 0 else float('+inf'),
            _levenshtein_damerau_distance(s1=s1[0:-1], s2=s2) + 1 if len(s1) > 0 else float('+inf'),
            _levenshtein_damerau_distance(s1=s1, s2=s2[0:-1]) + 1 if len(s2) > 0 else float('+inf'),
            _levenshtein_damerau_distance(s1=s1[0:-1], s2=s2[0:-1]) + (s1[-1] != s2[-1]) if len(s1) > 0 and len(s2) > 0 else float('+inf'),
            _levenshtein_damerau_distance(s1=s1[0:-2], s2=s2[0:-2]) + 1 if len(s1) > 1 and len(s2) > 1 and (s1[-1] == s2[-2]) and (s1[-2] == s2[-1]) else float('+inf'),
        )

        d_t_lengths_levenshtein_damerau_distance[(len(s1), len(s2))] = res

        return res
    
    _ = _levenshtein_damerau_distance(s1=s1, s2=s2)

    np_matrix = np.zeros((len(s1)+1, len(s2)+1))

    for i in range(0, len(s1)+1):
        for j in range(0, len(s2)+1):
            np_matrix[i, j] = d_t_lengths_levenshtein_damerau_distance[(i, j)]

    M = np_matrix

	#### Край на Вашия код
	#############################################################################

    return M

def editWeight(s1, s2, Weight):
	#### функцията editWeight намира теглото между два низа
	#### вход: низовете s1 и s2, както и речник Weight, съдържащ теглото на всяка от елементарните редакции 
	#### изход: минималната сума от теглата на елементарните редакции, необходими да се получи от единия низ другия

	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 15-30 реда

	d_t_lengths_weight = {}

	def _get_min_weight(s1: str, s2: str) -> int:
		cached_res = d_t_lengths_weight.get((len(s1), len(s2)))

		if cached_res is not None:
			return cached_res
		
		res = min(
			0 if len(s1) == 0 and len(s2) == 0 else float('+inf'),
			_get_min_weight(s1=s1[0:-1], s2=s2) + Weight[(s1[-1], "")] if len(s1) > 0 else float('+inf'),
			_get_min_weight(s1=s1, s2=s2[0:-1]) + Weight[("", s2[-1])] if len(s2) > 0 else float('+inf'),
			_get_min_weight(s1=s1[0:-1], s2=s2[0:-1]) + Weight[(s1[-1], s2[-1])] if len(s1) > 0 and len(s2) > 0 else float('+inf'),
			_get_min_weight(s1=s1[0:-2], s2=s2[0:-2]) + Weight[(s1[-2:], s2[-2:])] if len(s1) > 1 and len(s2) > 1 and (s1[-1] == s2[-2]) and (s1[-2] == s2[-1]) else float('+inf'),
		)

		d_t_lengths_weight[(len(s1), len(s2))] = res

		return res
	
	return _get_min_weight(s1=s1, s2=s2)

	#### Край на Вашия код
	#############################################################################


def bestAlignment(s1, s2):
	#### функцията намира подравняване с минимално тегло между два низа 
	#### вход: 
	####     низовете s1 и s2
	#### изход: 
	####     списък от елементарни редакции, подравняващи s1 и s2 с минимално тегло


	M = editDistance(s1, s2)
	alignment = []
	
	#############################################################################
	#### УПЪТВАНЕ:
	#### За да намерите подравняване с минимално тегло следва да намерите път в матрицата M,
	#### започващ от последния елемент на матрицата -- M[len(s1),len(s2)] до елемента M[0,0]. 
	#### Всеки преход следва да съответства на елементарна редакция, която ни дава минимално
	#### тегло, съответстващо на избора за получаването на M[i,j] във функцията editDistance.
	#### Събирайки съответните елементарни редакции по пъта от M[len(s1),len(s2)] до M[0,0] 
	#### в обратен ред ще получим подравняване с минимално тегло между двата низа.
	#### Всяка елементарна редакция следва да се представи като двойка низове.
	#### ПРЕМЕР:
	#### bestAlignment('редакция','рдацкиа') = [('р','р'),('е',''),('д' 'д'),('а','а'),('кц','цк'),('и','и'),('я','а')]
	#### ВНИМАНИЕ:
	#### За някой двойки от думи може да съществува повече от едно подравняване с минимално тегло.
	#### Достатъчно е да изведете едно от подравняванията с минимално тегло.
	#############################################################################	
	
	#############################################################################	
	#### Начало на Вашия код. На мястото на pass се очакват 15-30 реда

	alignment = []

	i = len(s1)
	j = len(s2)

	i, j = len(s1), len(s2)

	while i > 0 or j > 0:
		current_value = M[i, j]

		if i > 0 and j > 0 and M[i - 1, j - 1] + (s1[i - 1] != s2[j - 1]) == current_value:
			alignment.append((s1[i - 1], s2[j - 1]))
			i -= 1
			j -= 1
		elif i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1] and M[i - 2, j - 2] + 1 == current_value:
			alignment.append((s1[i - 2:i], s2[j - 2:j]))
			i -= 2
			j -= 2
		elif i > 0 and M[i - 1, j] + 1 == current_value:
			alignment.append((s1[i - 1], ""))
			i -= 1
		elif j > 0 and M[i, j - 1] + 1 == current_value:
			alignment.append(("", s2[j - 1]))
			j -= 1

	alignment = alignment[::-1]
			
	#### Край на Вашия код
	#############################################################################
			
	return alignment

def trainWeights(corpus):
	#### Функцията editionWeights връща речник съдържащ теглото на всяка от елементарните редакции
	#### Функцията реализира статистика за честотата на елементарните редакции от корпус, състоящ се от двойки сгрешен низ и коригиран низ. Теглата са получени след оценка на вероятността за съответната грешка, използвайки принципа за максимално правдоподобие.
	#### Вход: Корпус от двойки сгрешен низ и коригиран низ
	#### изход: речник съдържащ теглото на всяка от елементарните редакции

	opCount = {}
	
	ids = subs = ins = dels = trs = 0
	for q,r in tqdm(corpus):
		alignment = bestAlignment(q,r)
		for op in alignment:
			if len(op[0]) == 1 and  len(op[1]) == 1 and op[0] == op[1]: ids += 1
			elif len(op[0]) == 1 and  len(op[1]) == 1: subs += 1
			elif len(op[0]) == 0 and  len(op[1]) == 1: ins += 1
			elif len(op[0]) == 1 and  len(op[1]) == 0: dels += 1
			elif len(op[0]) == 2 and  len(op[1]) == 2: trs += 1
	N = ids + subs + ins + dels + trs

	weight = {}
	for a in langmodel.alphabet:
		weight[(a,a)] = - math.log( ids / N )
		weight[(a,'')] = - math.log( dels / N )
		weight[('',a)] = - math.log( ins / N )
		for b in langmodel.alphabet:
			if a != b:
				weight[(a,b)] = - math.log( subs / N )
			weight[(a+b,b+a)] = - math.log( trs / N )
	return weight


def generateEdits(q):
	### помощната функция, generate_edits по зададена заявка генерира всички възможни редакции на разстояние едно от тази заявка.
	### Вход: заявка като низ q
	### Изход: Списък от низове на Левенщайн - Дамерау разстояние 1 от q
	###
	### В тази функция вероятно ще трябва да използвате азбука, която е дефинирана в langmodel.alphabet
	###
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 10-15 реда

	l_strings_at_distance_1_from_q = []

	l_characters = langmodel.alphabet

	for character in l_characters:
		for i in range(len(q)+1):
			try:
				if character != q[i]:
					string_at_distance_1_from_q = q[0:i] + character + q[i+1:]
					l_strings_at_distance_1_from_q.append(string_at_distance_1_from_q)
			except:
				pass

			string_at_distance_1_from_q = q[0:i] + character + q[i:]
			l_strings_at_distance_1_from_q.append(string_at_distance_1_from_q)
	
	for i in range(len(q)):
		string_at_distance_1_from_q = q[0:i] + q[i+1:]
		l_strings_at_distance_1_from_q.append(string_at_distance_1_from_q)

	for i in range(len(q)-1):
		if q[i] != q[i+1]:
			string_at_distance_1_from_q = q[0:i] + q[i+1] + q[i] + q[i+2:]
			l_strings_at_distance_1_from_q.append(string_at_distance_1_from_q)

	return l_strings_at_distance_1_from_q

	#### Край на Вашия код
	#############################################################################


def generateCandidates(query,dictionary):
	### Започва от заявката query и генерира всички низове НА РАЗСТОЯНИЕ <= 2, за да се получат кандидатите за корекция. Връщат се единствено кандидати, за които всички думи са в речника dictionary.
		
	### Вход:
	###	 Входен низ: query
	###	 Речник: dictionary

	### Изход:
	###	 Списък от низовете, които са кандидати за корекция
	
	def allWordsInDictionary(q):
		### Помощна функция, която връща истина, ако всички думи в заявката са в речника
		return all(w in dictionary for w in q.split())


	L=[]
	if allWordsInDictionary(query):
		L.append(query)
	for query1 in generateEdits(query):
		if allWordsInDictionary(query1):
			L.append(query1)
		for query2 in generateEdits(query1):
			if allWordsInDictionary(query2):
				L.append(query2)
	return L



def correctSpelling(r, model: langmodel.MarkovModel, weights, mu = 1.0, alpha = 0.9):
	### Комбинира вероятността от езиковия модел с вероятността за редактиране на кандидатите за корекция, генерирани от generate_candidates за намиране на най-вероятната желана (коригирана) заявка по дадената оригинална заявка query.
	###
	### Вход:
	###	    заявка: r,
	###	    езиков модел: model,
	###     речник съдържащ теглото на всяка от елементарните редакции: weights
	###	    тегло на езиковия модел: mu
	###	    коефициент за интерполация на езиковият модел: alpha
	### Изход: най-вероятната заявка


	### УПЪТВАНЕ:
	###    Удачно е да работите с логаритъм от вероятностите. Логаритъм от вероятността от езиковия модел може да получите като извикате метода model.sentenceLogProbability. Минус логаритъм от вероятността за редактиране може да получите като извикате функцията editWeight.
	#############################################################################
	#### Начало на Вашия код за основното тяло на функцията correct_spelling. На мястото на pass се очакват 3-10 реда

	l_dictionary = model.kgrams[()].keys()

	l_candidates = generateCandidates(query=r, dictionary=l_dictionary)

	l_t_candidate_probability = []

	for candidate in tqdm(l_candidates, desc=f"mu = "):
		correction_model_probability = -np.log(editWeight(s1=r, s2=candidate, Weight=weights))

		language_model_probability = model.sentenceLogProbability(s=candidate, alpha=alpha)
		combined_probability = correction_model_probability * (language_model_probability ** mu)

		l_t_candidate_probability.append((candidate, combined_probability))

	l_t_candidate_probability_sorted = sorted(l_t_candidate_probability, key=lambda x: x[1])

	try:
		return l_t_candidate_probability_sorted[-1][0]
	except:
		return None

	#### Край на Вашия код
	#############################################################################