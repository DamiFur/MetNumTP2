toParse = open("test1.expected", 'r')
output = open("test1.expected.proceced", 'w')

count = 0
discart = False
for line in toParse:
	if discart:
		if count < 9:
			count += 1
		else:
			count = 0
			discart = False
	else:
		if count < 49:
			output.write(str(line))
			count += 1
		else:
			output.write(str(line))
			count = 0
			discart = True