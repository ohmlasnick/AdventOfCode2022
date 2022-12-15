import numpy as np

### DAY 1 CODE ###

def parse_lsts(txt_fle):
	with open(txt_fle, "r") as myfile:
		string=[i.replace('\n', '') if i != '\n' else i.replace('\n', '|') for i in myfile.readlines()]
		lst_of_food = [[int(j) for j in i.split(' ')] for i in " ".join(string).split(' | ')]
		return lst_of_food

def chonky_elf(lst_of_elves):
	total_each_elf = sorted([sum(elf) for elf in lst_of_elves])
	top_elf = max([sum(elf) for elf in lst_of_elves])
	top_three_elves = total_each_elf[-1] + total_each_elf[-2] + total_each_elf[-3]
	return top_elf, top_three_elves

#print(chonky_elf(parse_lsts("input_Day1.txt")))

### DAY 2 CODE ###

def parse_rounds(txt_fle):
	with open(txt_fle, "r") as myfile:
		round_lsts = [i.replace('\n', '') for i in myfile.readlines()]
		return round_lsts

def points_a(rnd):
	point_dict = {'A X': 4, 'A Y': 8, 'A Z': 3,
	'B X': 1, 'B Y': 5, 'B Z': 9,
	'C X': 7, 'C Y': 2, 'C Z': 6}
	return point_dict[rnd]

def points_b(rnd):
	point_dict = {'A X': 3, 'A Y': 4, 'A Z': 8,
	'B X': 1, 'B Y': 5, 'B Z': 9,
	'C X': 2, 'C Y': 6, 'C Z': 7}
	return point_dict[rnd]

def total_points(round_lsts):
	return sum([points_a(rnd) for rnd in round_lsts]), sum([points_b(rnd) for rnd in round_lsts])

#print(total_points(parse_rounds("input_Day2.txt")))

### DAY 3 CODE ###

def priority(character):
	if character.islower():
		return ord(character) - 96
	else:
		return ord(character) - 38

def common_halves(strng):
	first, second = strng[:(len(strng) // 2)], strng[(len(strng) // 2):]
	return list(set(first).intersection(set(second)))[0]

def common_three(group_index, lst_of_items):
	first = lst_of_items[group_index]
	second = lst_of_items[group_index + 1]
	third = lst_of_items[group_index + 2]
	return list(set(first).intersection(set(second)).intersection(set(third)))[0]

def sum_priority(lst_of_items):
	return (sum([priority(common_halves(item)) for item in lst_of_items]), 
		sum([priority(common_three(3 * i, lst_of_items)) for i in range(len(lst_of_items) // 3)]))

#print(sum_priority(parse_rounds("input_Day3.txt")))

### DAY 4 CODE ###

def a_in_b(a, b):
	if ((int(b[0]) <= int(a[0])) and (int(a[1]) <= int(b[1]))):
		return 1
	elif ((int(a[0]) <= int(b[0])) and (int(b[1]) <= int(a[1]))):
		return 1
	else:
		return 0

def a_b_overlap(a, b):
	if a_in_b(a, b):
		return 1
	elif ((int(a[0]) <= int(b[0])) and (int(b[0]) <= int(a[1]))):
		return 1
	elif ((int(b[0]) <= int(a[0])) and (int(a[0]) <= int(b[1]))):
		return 1
	else:
		return 0

def count_ranges(pairs):
	return (sum([a_in_b(sets[0], sets[1]) for sets in pairs]),
		sum([a_b_overlap(sets[0], sets[1]) for sets in pairs]))

def create_pair_sets(lst_of_pairs):
	pairs = [[elf.split('-') for elf in p.split(',')] for p in lst_of_pairs]
	return pairs

#print(count_ranges(create_pair_sets(parse_rounds("input_Day4.txt"))))

### DAY 5 CODE ###

SHIP_CRATES_a = {'1': ['N', 'D', 'M', 'Q', 'B', 'P', 'Z'],
			'2': ['C', 'L', 'Z', 'Q', 'M', 'D', 'H', 'V'],
			'3': ['Q', 'H', 'R', 'D', 'V', 'F', 'Z', 'G'],
			'4': ['H', 'G', 'D', 'F', 'N'],
			'5': ['N', 'F', 'Q'],
			'6': ['D', 'Q', 'V', 'Z', 'F', 'B', 'T'],
			'7': ['Q', 'M', 'T', 'Z', 'D', 'V', 'S', 'H'],
			'8': ['M', 'G', 'F', 'P', 'N', 'Q'],
			'9': ['B', 'W', 'R', 'M']}

SHIP_CRATES_b = {'1': ['N', 'D', 'M', 'Q', 'B', 'P', 'Z'],
			'2': ['C', 'L', 'Z', 'Q', 'M', 'D', 'H', 'V'],
			'3': ['Q', 'H', 'R', 'D', 'V', 'F', 'Z', 'G'],
			'4': ['H', 'G', 'D', 'F', 'N'],
			'5': ['N', 'F', 'Q'],
			'6': ['D', 'Q', 'V', 'Z', 'F', 'B', 'T'],
			'7': ['Q', 'M', 'T', 'Z', 'D', 'V', 'S', 'H'],
			'8': ['M', 'G', 'F', 'P', 'N', 'Q'],
			'9': ['B', 'W', 'R', 'M']}

def move_boxes_a(instruction):
	i = instruction[0]
	stack_j, stack_k = instruction[1], instruction[2]
	while i > 0:
		crate = SHIP_CRATES_a[stack_j].pop()
		SHIP_CRATES_a[stack_k].append(crate)
		i = i - 1

def move_boxes_b(instruction):
	i = instruction[0]
	stack_j, stack_k = instruction[1], instruction[2]
	crates = []
	while i > 0:
		crates.append(SHIP_CRATES_b[stack_j].pop())
		i = i - 1
	crates.reverse()
	SHIP_CRATES_b[stack_k] = SHIP_CRATES_b[stack_k] + crates

def crane_operator(lst_of_instructs, version):
	for instruction in lst_of_instructs:
		instr_parse = instruction.split(" ")
		instruct_code = (int(instr_parse[1]), instr_parse[3], instr_parse[5])
		if version == 'a':
			move_boxes_a(instruct_code)
		else:
			move_boxes_b(instruct_code)
	if version == 'a':
		SHIP_CRATES = SHIP_CRATES_a
	else:
		SHIP_CRATES = SHIP_CRATES_b
	return SHIP_CRATES['1'][-1] + SHIP_CRATES['2'][-1] + SHIP_CRATES['3'][-1] + SHIP_CRATES['4'][-1] + SHIP_CRATES['5'][-1] + SHIP_CRATES['6'][-1] + SHIP_CRATES['7'][-1] + SHIP_CRATES['8'][-1] + SHIP_CRATES['9'][-1]

print(crane_operator(parse_rounds("input_Day5.txt")[10:], 'a'))
print(crane_operator(parse_rounds("input_Day5.txt")[10:], 'b'))

### DAY 6 CODE ###



### DAY 7 CODE ###



### DAY 8 CODE ###



### DAY 9 CODE ###



### DAY 10 CODE ###



### DAY 11 CODE ###



### DAY 12 CODE ###



### DAY 13 CODE ###



### DAY 14 CODE ###