import numpy as np
from time import sleep
from math import ceil, floor
import copy
import re
from functools import reduce
import inspect
from queue import Queue
import itertools

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

#print(max_scenic_score(load_array("input_Day8.txt")))

### DAY 9 CODE ###

class Head:

	def __init__(self, pos):
		self.name = 'Head'
		self.pos = pos
		self.tails = []

	def move_head(self, direc, steps):
		HEAD_POS = self.pos
		for i in range(steps):
			if direc == 'U':
				HEAD_POS[0] += 1
			elif direc == 'D':
				HEAD_POS[0] -= 1
			elif direc == 'R':
				HEAD_POS[1] += 1
			else:
				HEAD_POS[1] -= 1
			for tail in self.tails:
				tail.check_tail()

class Tail:

	def __init__(self, name, pos, head):
  		self.pos = pos
  		self.name = name
  		self.head = head
  		self.visited = []

	def touching(self):
		HEAD_POS = self.head.pos
		row_diff = abs(HEAD_POS[0] - self.pos[0])
		col_diff = abs(HEAD_POS[1] - self.pos[1])
		dist = row_diff + col_diff
		return (HEAD_POS[0] == self.pos[0] and dist <= 1) or (HEAD_POS[1] == self.pos[1] and dist <= 1) or (row_diff == 1 and col_diff == 1)

	def is_diag_up_left(self):
		HEAD_POS = self.head.pos
		return HEAD_POS[0] > self.pos[0] and HEAD_POS[1] < self.pos[1]

	def is_diag_up_right(self):
		HEAD_POS = self.head.pos
		return HEAD_POS[0] > self.pos[0] and HEAD_POS[1] > self.pos[1]

	def is_diag_down_left(self):
		HEAD_POS = self.head.pos
		return HEAD_POS[0] < self.pos[0] and HEAD_POS[1] < self.pos[1]

	def is_diag_down_right(self):
		HEAD_POS = self.head.pos
		return HEAD_POS[0] < self.pos[0] and HEAD_POS[1] > self.pos[1]

	def is_top(self):
		HEAD_POS = self.head.pos
		return HEAD_POS[1] == self.pos[1] and HEAD_POS[0] > self.pos[0]

	def is_bottom(self):
		HEAD_POS = self.head.pos
		return HEAD_POS[1] == self.pos[1] and HEAD_POS[0] < self.pos[0]

	def is_left(self):
		HEAD_POS = self.head.pos
		return HEAD_POS[0] == self.pos[0] and HEAD_POS[1] < self.pos[1]

	def is_right(self):
		HEAD_POS = self.head.pos
		return HEAD_POS[0] == self.pos[0] and HEAD_POS[1] > self.pos[1]

	def check_tail(self):
		TAIL_POS = self.pos
		if not self.touching():
			if self.is_diag_up_left():
				TAIL_POS[0] += 1
				TAIL_POS[1] -= 1
			elif self.is_diag_up_right():
				TAIL_POS[0] += 1
				TAIL_POS[1] += 1
			elif self.is_diag_down_left():
				TAIL_POS[0] -= 1
				TAIL_POS[1] -= 1
			elif self.is_diag_down_right():
				TAIL_POS[0] -= 1
				TAIL_POS[1] += 1
			elif self.is_top():
				TAIL_POS[0] += 1
			elif self.is_bottom():
				TAIL_POS[0] -= 1
			elif self.is_left():
				TAIL_POS[1] -= 1
			elif self.is_right():
				TAIL_POS[1] += 1
		self.visited += [str(TAIL_POS[0])+str(TAIL_POS[1])]

def pull_rope(lst_of_moves):
	HEAD = Head([0,0])
	HEAD.tails += [Tail('1', [0,0], HEAD)]
	for t in range(2,10):
		HEAD.tails += [Tail(str(t), [0,0], HEAD.tails[t-2])]
	for move in lst_of_moves:
		direc, steps = move.split(' ')
		HEAD.move_head(direc, int(steps))
	return len(set(HEAD.tails[8].visited))

#print(pull_rope(parse_rounds("input_Day9.txt")))

### DAY 10 CODE ###

class Clock:

	def __init__(self, cpu_display):
		self.cycle = 0
		self.cpu_display = cpu_display

	def tick(self):
		# every time it ticks, needs to check if row has changed
		self.cpu_display.draw_pixel(self.cycle)
		self.cycle += 1
		self.cpu_display.col += 1
		if self.cpu_display.col > 39:
			self.cpu_display.row += 1
			self.cpu_display.col = 0

class Register:

	def __init__(self):
		self.value = 1

class CPU:

	def __init__(self):
		self.name = 'CPU'
		self.clock = Clock(self)
		self.register = Register()
		self.sprite_pixels = [0, 1, 2]
		self.row = 0
		self.col = 0
		self.pixels = np.reshape(np.array(['.'] * 240), (6,40))

	def signal_strength(self):
		return self.clock.cycle * self.register.value

	def addx(self, val):
		self.register.value += val
		self.sprite_pixels = [self.register.value - 1, self.register.value, self.register.value + 1]

	def render(self):
		for row in self.pixels:
			print("".join(row))

	def draw_pixel(self, cycle):
		for s in self.sprite_pixels:
			if s == self.col:
				self.pixels[self.row][self.col] = '#'

def execute_CPU_coms(lst_of_cpu_coms):
	signal_strengths = []
	cpu = CPU()
	cpu.render()
	for com in lst_of_cpu_coms:
		if 'addx' in com:
			cpu.clock.tick()
			signal_strengths += [cpu.signal_strength()]
			cpu.clock.tick()
			signal_strengths += [cpu.signal_strength()]
			cpu.addx(int(com.split(' ')[1]))
		elif 'noop' in com:
			cpu.clock.tick()
			signal_strengths += [cpu.signal_strength()]
	filtered = [signal_strengths[19],signal_strengths[59],signal_strengths[99],
				signal_strengths[139],signal_strengths[179],signal_strengths[219]]
	cpu.render()
	return sum(filtered)

#print(execute_CPU_coms(parse_rounds("input_Day10.txt")))

### DAY 11 CODE ###

def greatest_comm_d(a, b):
    if b == 0:
        return a
    else:
    	return greatest_comm_d(b, a % b)

def least_comm_mult(a, b):
    return (a * b) // greatest_comm_d(a, b)

class Monkey:

	def __init__(self, name, items, op, test):
		self.name = name
		self.items = items
		self.op = op
		self.test = test
		self.true_monkey = None
		self.false_monkey = None
		self.to_be_deleted = []
		self.num_inspections = 0

	def throw_item(self, item, monkey):
		monkey.items += [copy.deepcopy(item)]
		ind = self.items.index(item)
		self.to_be_deleted += [ind]

	def inspect(self, item, least_common):
		item.worry_level = self.op(item.worry_level) % least_common
		self.num_inspections += 1

	def test_func(self, item):
		return item.worry_level % self.test == 0

	def test_worry(self, item):
		if self.test_func(item):
			self.throw_item(item, self.true_monkey)
		else:
			self.throw_item(item, self.false_monkey)

	def delete_thrown_items(self):
		self.items = [v for i, v in enumerate(self.items) if i not in self.to_be_deleted]

class Item:

	def __init__(self, worry_level):
		self.worry_level = worry_level

def create_monkeys():
	monkeys = [Monkey('0', [Item(99),Item(63),Item(76),Item(93),Item(54),Item(73)], lambda x: x*11, 2),
				Monkey('1', [Item(91),Item(60),Item(97),Item(54)], lambda x: x+1, 17),
				Monkey('2', [Item(65)], lambda x: x+7, 7),
				Monkey('3', [Item(84),Item(55)], lambda x: x+3, 11),
				Monkey('4', [Item(86),Item(63),Item(79),Item(54),Item(83)], lambda x: x*x, 19),
				Monkey('5', [Item(96),Item(67),Item(56),Item(95),Item(64),Item(69),Item(96)], lambda x: x+4, 5),
				Monkey('6', [Item(66),Item(94),Item(70),Item(93),Item(72),Item(67),Item(88),Item(51)], lambda x: x*5, 13),
				Monkey('7', [Item(59),Item(59),Item(74)], lambda x: x+8, 3)]
	monkeys[0].true_monkey = monkeys[7]
	monkeys[0].false_monkey = monkeys[1]
	monkeys[1].true_monkey = monkeys[3]
	monkeys[1].false_monkey = monkeys[2]
	monkeys[2].true_monkey = monkeys[6]
	monkeys[2].false_monkey = monkeys[5]
	monkeys[3].true_monkey = monkeys[2]
	monkeys[3].false_monkey = monkeys[6]
	monkeys[4].true_monkey = monkeys[7]
	monkeys[4].false_monkey = monkeys[0]
	monkeys[5].true_monkey = monkeys[4]
	monkeys[5].false_monkey = monkeys[0]
	monkeys[6].true_monkey = monkeys[4]
	monkeys[6].false_monkey = monkeys[5]
	monkeys[7].true_monkey = monkeys[1]
	monkeys[7].false_monkey = monkeys[3]
	return monkeys

def do_n_rounds(monkeys, n):
	least_common = reduce(least_comm_mult, [m.test for m in monkeys])
	for i in range(n):
		for m in monkeys:
			for item in m.items:
				m.inspect(item, least_common)
				#item.worry_level = floor(item.worry_level / 3.)
				m.test_worry(item)
			m.delete_thrown_items()
	items_counted = []
	for m in monkeys:
		items_counted += [m.num_inspections]
	items_counted = sorted(items_counted, reverse=True)
	monkey_business = items_counted[0] * items_counted[1]
	return monkey_business

#print(do_n_rounds(create_monkeys(), 10000))

### DAY 12 CODE ###

### DAY 13 CODE ###

def in_right_order(p1_lst, p2_lst, level=0):
	for left, right in itertools.zip_longest(p1_lst, p2_lst):
		#print(' ' * level, 'Compare ', left, ' to ', right)
		if type(left) == int and type(right) == int:
			# Can stop at first pair of vals that are not equal
			if left < right:
				return True
			elif right < left:
				return False
			else:
				continue
		if left == None and right != None:
			return True
		if right == None and left != None:
			return False
		if type(left) == list and type(right) == list:
			# Retry by comparing values in left, right
			do_next = in_right_order(left, right, level+1)
		if type(left) == int and type(right) == list:
			# Retry with int as list
			do_next = in_right_order([left], right, level+1)
		if type(right) == int and type(left) == list:
			# Retry with int as list
			do_next = in_right_order(left, [right], level+1)
		if do_next == None:
			continue
		else:
			return do_next

def compare_all_packets(fle):
	import json
	pairs = [p for p in parse_rounds(fle) if p.strip() != '']
	right_order = []
	i = 0
	p_ind = 1
	end = (len(pairs) - 1)
	while i < end:
		p1_lst = json.loads(pairs[i])
		p2_lst = json.loads(pairs[i+1])
		ordered = in_right_order(p1_lst, p2_lst)
		if ordered:
			right_order += [p_ind]
		p_ind += 1
		i += 2
	return sum(right_order)

def custom_sort_decode(fle):
	import json
	pairs = [json.loads(p) for p in parse_rounds(fle) if p.strip() != '']
	sorted_pairs = []
	for p in pairs:
		yet_inserted = False
		print("Inserting ", p)
		if len(sorted_pairs) == 0:
			sorted_pairs += [p]
			continue
		for i in range(len(sorted_pairs)):
			sort_p = sorted_pairs[i]
			print("     Comparing to ", sort_p)
			smaller = in_right_order(p, sort_p)
			print("          Smaller: ", smaller)
			if smaller:
				sorted_pairs.insert(i, p)
				yet_inserted = True
				break
		if not yet_inserted:
			sorted_pairs += [p]
		print("          New sorted: ", sorted_pairs)
	for p in sorted_pairs:
		print(p)
	decoder_ind1, decoder_ind2 = sorted_pairs.index([[2]]), sorted_pairs.index([[6]])
	print(decoder_ind1, decoder_ind2)
	return (decoder_ind1+1) * (decoder_ind2+1)

#print(compare_all_packets("input_Day13.txt"))
#print(custom_sort_decode("input_Day13.txt"))

### DAY 14 CODE ###

class Map():

	def __init__(self, floor=False):
		self.floor = floor
		self.rocks = parse_rocks(parse_rounds("Day14_example.txt"))
		self.scan_map = [['.','.'],
						['.','.']]
		self.x = len(self.scan_map)
		self.y = len(self.scan_map[0])
		for rock in self.rocks:
			for line in rock.lines:
				segments = line.get_intermed()
				for seg in segments:
					if seg[1] >= self.x or seg[0] >= self.y:
						print('expanding map...')
						self.expand_map(seg[1], seg[0])
					self.scan_map[seg[1]][seg[0]] = '#'
		try:
			self.scan_map[0][500] = '+'
		except:
			self.expand_map(0, 500)
		print('shrinking map...')
		self.shrink_map()
		# Add empty 1st and last cols so sand can fall into void
		self.expand_map(0, len(self.scan_map[0]))
		self.expand_map(0, -1)
		if floor:
			self.add_floor()
		self.show_map()
		print('Size: ', len(self.scan_map), ' x ', len(self.scan_map[0]))
		#print(self.x, self.y)

	def count_sand(self):
		count = 0
		for row in self.scan_map:
			for cell in row:
				if cell == 'o' or cell == '+':
					count += 1
		return count

	def show_map(self):
		for row in self.scan_map:
			print("".join(row))

	def expand_map(self, new_x, new_y):
		while new_x >= self.x:
			self.scan_map += [['.'] * self.y]
			self.x += 1
		while new_y >= self.y:
			for row in range(self.x):
				self.scan_map[row].append('.')
			self.y += 1
		if new_y == -1:
			for row in range(self.x):
				self.scan_map[row] = ['.'] + self.scan_map[row]
			self.y += 1

	def shrink_map(self):
		# Get rid of unnecessary rows and columns (all dots)
		num_rows_remove = 0
		transposed_scan = list(zip(*self.scan_map))
		for i in range(self.x):
			row = self.scan_map[i]
			if '#' not in row and '+' not in row:
				num_rows_remove += 1
			else:
				break
		num_cols_remove = 0
		for j in range(self.y):
			col = transposed_scan[j]
			if '#' not in col and '+' not in col:
				num_cols_remove += 1
			else:
				break
		self.scan_map = self.scan_map[num_rows_remove:]
		transposed_scan = [list(z) for z in zip(*self.scan_map)][num_cols_remove:]
		self.scan_map = [list(z) for z in zip(*transposed_scan)]
		self.x = len(self.scan_map)
		self.y = len(self.scan_map[0])

	def add_floor(self):
		self.expand_map((self.x + 1), 0)
		self.scan_map[self.x - 1] = ['#'] * self.y

class Rock():

	def __init__(self, name, lines):
		self.name = name
		self.lines = lines

class Line():

	def __init__(self, start, end):
		make_coords = lambda strng: [int(s) for s in strng.split(',')]
		self.coords = [make_coords(start), make_coords(end)]

	def get_intermed(self):
		start, end = self.coords
		diff_x = end[0] - start[0]
		diff_y = end[1] - start[1]
		between = []
		if diff_x == 0:
			sign = diff_y // abs(diff_y)
			for i in range(1, abs(diff_y)):
				between += [[start[0], start[1] + (sign * i)]]
		else:
			sign = diff_x // abs(diff_x)
			for i in range(1, abs(diff_x)):
				between += [[start[0] + (sign * i), start[1]]]
		return [start] + between + [end]

class Sand():

	def __init__(self, rock_map):
		self.pos_x = 0
		self.pos_y = rock_map.scan_map[0].index('+')
		self.map = rock_map

	def free_below(self):
		if self.on_floor():
			return False
		return self.map.scan_map[self.pos_x + 1][self.pos_y] == '.'

	def free_left(self):
		# If there's no space, you can expand walls
		if self.on_floor():
			return False
		if self.pos_y == 0:
			# but only if right is also blocked
			if not self.free_right():
				self.map.expand_map(0, -1)
				if self.map.floor:
					self.map.scan_map[self.map.x - 1] = ['#'] * self.map.y
			else:
				return False
		return self.map.scan_map[self.pos_x + 1][self.pos_y - 1] == '.'

	def free_right(self):
		# If there's no space, you can expand walls
		if self.on_floor():
			return False
		try:
			return self.map.scan_map[self.pos_x + 1][self.pos_y + 1] == '.'
		except:
			self.map.expand_map(0, self.map.y)
			if self.map.floor:
				self.map.scan_map[self.map.x - 1] = ['#'] * self.map.y
			return self.map.scan_map[self.pos_x + 1][self.pos_y + 1] == '.'

	def move(self, direc):
		# free previous position if not origin
		if not self.map.scan_map[self.pos_x][self.pos_y] == '+':
			self.map.scan_map[self.pos_x][self.pos_y] = '.'
		self.pos_x += 1
		if direc == 'left-diag':
			self.pos_y -= 1
		elif direc == 'right-diag':
			self.pos_y += 1
		# move sand to new position
		self.map.scan_map[self.pos_x][self.pos_y] = 'o'

	def is_settled(self):
		return self.on_floor() or ((not self.free_below()) and (not self.free_left()) and (not self.free_right()))

	def on_floor(self):
		if self.map.floor:
			return (self.pos_x + 1) == (self.map.x - 1)
		else:
			return False

	def in_void(self):
		# if true, can return the total number of settled sands
		if self.map.floor:
			return self.is_settled() and self.pos_x == 0 and self.pos_y == self.map.scan_map[0].index('+')
		return self.pos_x == (len(self.map.scan_map) - 1)

def parse_rocks(rocks):
	all_rocks = []
	i = 1
	for rock in rocks:
		lines = rock.split(' -> ')
		line_objs = []
		for l in range(len(lines) - 1):
			obj = Line(lines[l], lines[l + 1])
			line_objs += [obj]
		all_rocks += [Rock(str(i), line_objs)]
		i += 1
	return all_rocks

def drop_sand():
	my_map = Map(floor=True)
	while True:
		# create new piece of sand
		sand = Sand(my_map)
		while (not sand.in_void()) and (not sand.is_settled()):
			sleep(0.03)
			if sand.free_below():
				sand.move('down')
			elif sand.free_left():
				sand.move('left-diag')
			elif sand.free_right():
				sand.move('right-diag')
			sand.map.show_map()
		if sand.in_void():
			return sand.map.count_sand()

#print(drop_sand())

### DAY 15 CODE ###