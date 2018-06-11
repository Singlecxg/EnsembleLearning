#coding=utf-8

WordDict = {}

f = open('Test_features.txt','r')
features = f.readlines()
f.close()
for i in features:
	feature = i.decode('utf-8','ignore').split()
	WordDict = dict([(w,1) for w in feature])
print(WordDict)

# fp = open('Test_validation_review.txt', 'r')
fp = open('Test_train.txt', 'r')
lines = fp.readlines()
f.close()

# fw = open('Test_validation_review_vector.txt','w')
fw = open('Test_train_vector.txt', 'w')

cnt = 0
for line in lines:
	cnt += 1
	if cnt > 30000:
		break
	# data = line.decode('utf-8','ignore').split(',', 1)[1].split()
	rank = line.decode('utf-8','ignore').split()[0]
	data = line.decode('utf-8','ignore').split()[1:]
	for word in data:
		tmp = WordDict.get(word)
		if tmp != None:
			WordDict[word] = tmp + 1
	for key in WordDict: 
		fw.write(str(WordDict[key]))
		fw.write(' ')
		WordDict[key] = 1
	fw.write(rank)
	fw.write('\n')

fw.close()