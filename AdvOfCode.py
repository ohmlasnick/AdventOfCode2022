import numpy as np
from time import sleep

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

#print(crane_operator(parse_rounds("input_Day5.txt")[10:], 'a'))
#print(crane_operator(parse_rounds("input_Day5.txt")[10:], 'b'))

### DAY 6 CODE ###

def parse_stream(txt_fle):
	with open(txt_fle, "r") as myfile:
		stream = myfile.read().strip()
		return stream

def all_diff(strng):
	settified = set(list(strng))
	return len(strng) == len(settified)

def start_of_packet(stream):
	for i in range(3, len(stream)):
		if all_diff(stream[(i - 3):(i + 1)]):
			break
	return i + 1

def start_of_message(stream):
	for i in range(13, len(stream)):
		if all_diff(stream[(i - 13):(i + 1)]):
			break
	return i + 1

#print(start_of_packet(parse_stream("input_Day6.txt")), start_of_message(parse_stream("input_Day6.txt")))

### DAY 7 CODE ###

class Direct:
  def __init__(self, name, parent):
  	self.name = name
  	self.parent = parent
  	self.children = []
  	self.files = []

  def get_size(self):
  	size = 0
  	for f in self.files:
  		size += int(f.split(' ')[0])
  	for c in self.children:
  		size += c.get_size()
  	return size

  def child_names(self):
  	return [c.name for c in self.children]

ROOT = Direct('/', None)
CURR_DIR = ROOT

def add_dir(dir_name):
	global CURR_DIR, ROOT
	if dir_name not in CURR_DIR.child_names():
		CURR_DIR.children += [Direct(dir_name, CURR_DIR)]

def process_cd(command):
	global CURR_DIR, ROOT
	dir_name = command.split(' ')[2]
	if command == '$ cd /':
		CURR_DIR = ROOT
	elif command == '$ cd ..':
		if CURR_DIR != ROOT:
			CURR_DIR = CURR_DIR.parent
	elif '$ cd ' in command:
		add_dir(dir_name)
		ind = CURR_DIR.child_names().index(dir_name)
		CURR_DIR = CURR_DIR.children[ind]

def process_dirs_fles(command):
	global CURR_DIR, ROOT
	com1, com2 = command.split(' ')
	if com1 == 'dir':
		add_dir(com2)
	else:
		CURR_DIR.files += [command]

def run_commands(lst_of_commands):
	global CURR_DIR, ROOT
	for command in lst_of_commands:
		if '$ cd ' in command:
			process_cd(command)
		elif command == '$ ls':
			continue
		else:
			process_dirs_fles(command)

def get_sizes(root_dir):
	sizes = [root_dir.get_size()]
	for child in root_dir.children:
		sizes += get_sizes(child)
	return sizes

def sum_sizes():
	sizes_less_100000 = [s for s in get_sizes(ROOT) if s <= 100000]
	return sum(sizes_less_100000)

def size_smallest_dir(root_dir, min_needed):
	sizes_more_than = [s for s in get_sizes(root_dir) if s >= min_needed]
	smallest = min(sizes_more_than)
	return smallest

#run_commands(parse_rounds("input_Day7.txt"))
#print(size_smallest_dir(ROOT, 30000000 - (70000000 - ROOT.get_size())))

### DAY 8 CODE ###

### Part a ###
def load_array(txt_fle):
	with open(txt_fle, "r") as myfile:
		round_lsts = [[int(j) for j in list(i.replace('\n', ''))] for i in myfile.readlines()]
		return np.array(round_lsts)

def blocked_left(i, j, mat):
	for neighbor in mat[i][:j]:
		if neighbor >= mat[i][j]:
			return True
	return False

def blocked_right(i, j, mat):
	for neighbor in mat[i][j+1:]:
		if neighbor >= mat[i][j]:
			return True
	return False

def blocked_top(i, j, mat):
	for neighbor in np.transpose(mat)[j][:i]:
		if neighbor >= np.transpose(mat)[j][i]:
			return True
	return False

def blocked_bottom(i, j, mat):
	for neighbor in np.transpose(mat)[j][i+1:]:
		if neighbor >= np.transpose(mat)[j][i]:
			return True
	return False

def blocked(i, j, mat):
	return blocked_left(i, j, mat) and blocked_right(i, j, mat) and blocked_top(i, j, mat) and blocked_bottom(i, j, mat)

def count_visible(mat):
	visible_mat = np.ones(mat.shape)
	for i in range(1, mat.shape[0] - 1): # row
		for j in range(1, mat.shape[1] - 1): # col
			if blocked(i, j, mat):
				visible_mat[i][j] = 0 
	return np.sum(visible_mat)


############################## Part b ##############################
def view_left(i, j, mat):
	neighbors = np.flip(mat[i][:j])
	for n in range(len(neighbors)):
		if neighbors[n] >= mat[i][j]:
			break
	return n+1

def view_right(i, j, mat):
	neighbors = mat[i][j+1:]
	for n in range(len(neighbors)):
		if neighbors[n] >= mat[i][j]:
			break
	return n+1

def view_top(i, j, mat):
	neighbors = np.flip(np.transpose(mat)[j][:i])
	for n in range(len(neighbors)):
		if neighbors[n] >= np.transpose(mat)[j][i]:
			break
	return n+1

def view_bottom(i, j, mat):
	neighbors = np.transpose(mat)[j][i+1:]
	for n in range(len(neighbors)):
		if neighbors[n] >= np.transpose(mat)[j][i]:
			break
	return n+1

def max_scenic_score(mat):
	scenic_scores = np.zeros(mat.shape)
	for i in range(1, mat.shape[0] - 1): # row
		for j in range(1, mat.shape[1] - 1): # col
			scenic_scores[i][j] = view_top(i, j, mat) * view_bottom(i, j, mat) * view_right(i, j, mat) * view_left(i, j, mat)
	return np.amax(scenic_scores)

print(max_scenic_score(load_array("input_Day8.txt")))

### DAY 9 CODE ###



### DAY 10 CODE ###



### DAY 11 CODE ###



### DAY 12 CODE ###



### DAY 13 CODE ###



### DAY 14 CODE ###