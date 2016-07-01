from PIL import Image, ImageDraw, ImageFont
import json
import textwrap
import os
from operator import itemgetter
import itertools
import numpy as np

# colors for captions
colors = [ (255,0,0), (0,255,0), (0,0,255), (255, 255, 0), (255,0,255), (0, 255, 255) ]

# for comparing captions by overlap of words (minus stopwords).
# there is a much better way to do this
stopwords=["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours	ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]


def resizeResultsToRaw(x, y):
	xR = x * float(wRaw) / wResults
	yR = y * float(hRaw) / hResults
	return (xR, yR)

def rectArea(r1, r2):
	dx = min(r1[0]+r1[2], r2[0]+r2[2]) - max(r1[0], r2[0])
	dy = min(r1[1]+r1[3], r2[1]+r2[3]) - max(r1[1], r2[1])
	if (dx>=0) and (dy>=0):
		return dx*dy
	else:
		return 0

def rectOverlap(r1, r2):
	w = max(r1[0]+r1[2], r2[0]+r2[2]) - min(r1[0], r2[0])
	h = max(r1[1]+r1[3], r2[1]+r2[3]) - min(r1[1], r2[1])
	totalArea = w * h
	intersectArea = rectArea(r1, r2)
	return float(intersectArea) / totalArea

def getCaptionOverlap(cap1, cap2):
	c1 = [c for c in cap1.split(' ') if c not in stopwords]
	c2 = [c for c in cap2.split(' ') if c not in stopwords]
	overlap = float(len([c for c in c1 if c in c2])) / len(list(set(c1+c2)))
	return overlap

def getNumCaptions(imageName):
	idxAll = range(len(data["results"]))
	idxP = [i for i in idxAll if data["results"][i]["img_name"]  == unicode(imageName)][0]
	return len(data["results"][idxP]["scores"])

def getCaptions(imageName, idxC):
	idxAll = range(len(data["results"]))
	idxP = [i for i in idxAll if data["results"][i]["img_name"]  == unicode(imageName)][0]
	name = data["results"][idxP]["img_name"]
	score = data["results"][idxP]["scores"][idxC]
	caption = data["results"][idxP]["captions"][idxC]
	rectResults = [data["results"][idxP]["boxes"][idxC][i] for i in range(4)]
	xR, yR = resizeResultsToRaw(rectResults[0], rectResults[1])
	wR, hR = resizeResultsToRaw(rectResults[2], rectResults[3])
	rect = [xR, yR, wR, hR]
	return (score, caption, rect)

def getAllCaptions(frame1, frame2):
	captions = []
	for k in range(frame1, frame2):
		if k % 10 == 0:	print "got %d"%k
		imageName = "frame%04d.png" % k
		n = min(len(colors), getNumCaptions(imageName))
		for i in range(n):
			score, caption, rect = getCaptions(imageName, i)
			captions.append({"captions":[caption], "scores":[score], "rects":[rect], "frames":[k]})
	captions = sorted(captions, key=itemgetter("scores"))
	captions.reverse()
	return captions

def mergeCaptions(captions):
	# sort captions into hashmap whose key is frame #
	fcaptions = {}
	num=0
	for i,c in enumerate(captions):
		for f in c['frames']:
			if f in fcaptions:
				fcaptions[f].append(i)
			else:
				fcaptions[f] = [i]
	# using above hashmap, group all captions that need to be compared
	# i.e. all pairs of captions whose frame difference < maxframediff
	compare = {}
	for f1 in range(frame1,frame2):
		if f1 not in fcaptions:
			continue
		captions1 = list(set(fcaptions[f1]))
		captions2 = []
		for f2 in range(f1+1,min(frame2-1,f1+1+maxframediff)):
			if f2 in fcaptions:
				captions2 += list(set(fcaptions[f2]))
		for c1 in captions1:
			for c2 in captions2:
				if c1==c2:	continue
				idx1 = min(c1,c2)
				idx2 = max(c1,c2)
				if idx1 not in compare:
					compare[idx1] = [idx2]
				elif idx2 not in compare[idx1]:
					compare[idx1].append(idx2)
	# filter compare down to list of candidate pairs to merge.
	# candidates must overlap 50% in rectangles, have overlapping
	# text and be within maxframediff captions
	candidates = []
	for i1 in compare:
		for i2 in compare[i1]:
			c1 = captions[i1]
			c2 = captions[i2]
			sameCaption = True
			for cap1 in c1['captions']:
				for cap2 in c2['captions']:
					capOverlap = getCaptionOverlap(cap1, cap2)
					if capOverlap <= 0.5:
						sameCaption = False
			frameDiff = 1e8
			for f1 in c1['frames']:
				for f2 in c2['frames']:
					if abs(f1-f2) < frameDiff:
						frameDiff = abs(f1-f2) 
			maxOverlap = 0
			for r1 in c1['rects']:
				for r2 in c2['rects']:
					maxOverlap = max(maxOverlap, rectOverlap(r1,r2))
			overlap = maxOverlap
			# calculate score and append to candidates if > 0
			score = sameCaption * (frameDiff < maxframediff) * (overlap > 0.5)  * overlap
			if score > 0:
				candidates.append({"idx":[i1,i2], "score":score})
	# sort the candidates by score
	candidates = sorted(candidates, key=itemgetter("score"))
	candidates.reverse()
	# go down list of candidates and decide to merge if neither 
	# individual caption has been merged yet
	mergedIdx = []
	toMerge = []
	for c in candidates:
		if c['idx'][0] not in mergedIdx and c['idx'][1] not in mergedIdx:
			mergedIdx.append(c['idx'][0])
			mergedIdx.append(c['idx'][1])
			toMerge.append(c['idx'])
	# now merge the accepted ones into newCaptions, then set that to captions
	# and add remaining captions (which weren't candidates) back to it
	newCaptions = []
	for m in toMerge:	
		caption_ = captions[m[0]]['captions'] + captions[m[1]]['captions']
		scores = captions[m[0]]['scores'] + captions[m[1]]['scores']
		frames = captions[m[0]]['frames'] + captions[m[1]]['frames']
		rects = captions[m[0]]['rects'] + captions[m[1]]['rects']
		newCaptions.append({"captions":caption_, "scores":scores, "rects":rects, "frames":frames})
	for i,c in enumerate(captions):
		if i not in mergedIdx:
			newCaptions.append(c)
	captions = newCaptions
	return captions

# from a group of merged caption, pick the one which appears
# the most times
def getTopCaption(caption):
	uniqueCaps = set(caption['captions'])
	topCaption = ''
	topLen = -1
	for cap in uniqueCaps:		
		n = len([c0 for c0 in caption['captions'] if c0==cap])
		if n > topLen:
			topLen = n
			topCaption = cap
	return cap

def normalizeCaptionScores(captions):
	frameCount = np.zeros(frame2-frame1)
	for k in range(frame1, frame2):
		for i,c in enumerate(captions):
			f1 = min(c["frames"])
			f2 = max(c["frames"])
			if f2-f1 > 20:
				fm = 0
			else:
				fm = int(frameMargin * float(len(c["frames"])) / (f2-f1+1))
			if k >= f1-fm and k <= f2+fm:
				frameCount[k-frame1]+=1
	for i,c in enumerate(captions):
		captions[i]['score'] = np.sum(c["scores"]) / frameCount[k-frame1]
	return captions


# experimental but doesn't work very well so not used.
# supposed to smooth out moving rectagle of a caption over frames
def getRect(captions, idxc, k):
	frames = captions[idxc]['frames']
	rects = captions[idxc]['rects']
	idx = sorted(range(len(frames)), key=lambda k: frames[k])
	srects = [rects[i] for i in idx]
	sframes = [frames[i] for i in idx]
	idxt = [i for i in range(len(sframes)) if sframes[i] < k ]
	if len(idxt)==0:
		idxt = 0
	else:
		idxt = max(idxt)
	total = 0.0
	mult = []
	for idx in range(0, len(captions[idxc]['frames'])):
		if idx >= idxt:
			t = float((len(captions[idxc]['frames']) - idxt+1) - (idx-idxt)) / (len(captions[idxc]['frames']) - idxt)
		else:
			t = float(idx+1) / idxt
		#t = pow(t, 2.0) 
		total += t
		mult.append(t)
	mm = 0
	tRect = [0.0, 0.0, 0.0, 0.0]
	for idx in range(0, len(sframes)):
		m = mult[idx] / total
		zz = [r for r in srects[idx]]
		tRect = [tr + r*m for tr,r in zip(tRect,zz)]
	tRect2 = srects[idxt]
	return tRect


def drawCaption(draw, text, myFont, color, rect, thickness, dr, dc, dt, drawCenter):
	x, y, w, h = rect
	x, y = max(0, x), max(0, y)
	rectColor = (color[0], color[1], color[2], 100)
	rectWidth = max(8, int(0.8*w / draw.textsize("z", font=myFont)[0])) #20 #0.6 * w / 10
	if dr:
		draw.line((x, y, x+w, y), fill=rectColor, width=thickness)
		draw.line((x+w, y-0.5*thickness+1, x+w, y+h+0.5*thickness-1), fill=rectColor, width=thickness)
		draw.line((x+w, y+h, x, y+h), fill=rectColor, width=thickness)
		draw.line((x, y+h+0.5*thickness-1, x, y-0.5*thickness+1), fill=rectColor, width=thickness)
	textLines = textwrap.wrap(text, width=rectWidth)
	if drawCenter:
		yt = y + 0.5 * h - 0.5 * len(textLines) * draw.textsize("z", font=myFont)[1]
	else:
		yt = y + 3
	for i,line in enumerate(textLines):
		tw = draw.textsize(line, font=myFont)
		if drawCenter:
			tx1 = 0.5*(2*x+w) - 0.5*tw[0]
		else:
			tx1 = x + 4
		ty1 = yt - 2
		if drawCenter:
			tx2 = 0.5*(2*x+w) + 0.5*tw[0]
		else:
			tx2 = x + 4 + tw[0]
		ty2 = yt + tw[1] + 2
		if dc:
			draw.rectangle((tx1-5, ty1, tx2+5, ty2), fill=color)
		if dt:
			draw.text((tx1, yt), line, font=myFont, fill="#000000")
 		yt += (myFont.getsize(line)[1])


def drawFrame(k, captions, nCaptions, display):
	imageName = "frame%04d.png" % k
	imagePath = frameDir+"frame%04d.png" % k
	im = Image.open(imagePath)
	if im.width != wRaw or im.height != hRaw:
		im = im.resize((wRaw, hRaw), Image.ANTIALIAS)
	draw = ImageDraw.Draw(im) 
	num = 0
	for i,c in enumerate(captions[0:min(nCaptions,len(captions))]):
		f1 = min(c["frames"])
		f2 = max(c["frames"])
		#fm = frameMargin
		if f2-f1 > 20:
			fm = 0
		else:
			fm = int(frameMargin * float(len(c["frames"])) / (f2-f1+1))
		if k >= f1-fm and k <= f2+fm and num < 10:
			num+=1
			cRect = [np.mean([r[0] for r in c["rects"]]), np.mean([r[1] for r in c["rects"]]), np.mean([r[2] for r in c["rects"]]), np.mean([r[3] for r in c["rects"]])]
			#cRect = getRect(captions, i, k)
			topCaption = getTopCaption(c)
			drawCaption(draw, topCaption, myFont, colors[i%len(colors)], cRect, 5, True, False, False, drawCenter)
			drawCaption(draw, topCaption, myFont, colors[i%len(colors)], cRect, 5, False, True, False, drawCenter)
			drawCaption(draw, topCaption, myFont, colors[i%len(colors)], cRect, 5, False, False, True, drawCenter)
	if display:
		im.show()
	return im



####################################
# setup paraemters

# size of the images returned by densecap. it does 720 on largest side by default
wResults = 720
hResults = 405

# since densecap downsamples the images, we can use the original high-res frames to make the video once 
# the captions have been generated
wRaw = 1280
hRaw = 720

# font options for writing captions
fontSize = 20
fontPath = 'Arial.ttf' 

# this is the path to the json file returned by densecap
jsonPath = 'results.json'

# if True, write the caption in the middle of the rectangle, otherwise top-left
drawCenter = False

# where the frames of the original video are (set this)
frameDir = "originalFrames"

# where to put the captioned images (create this directory)
newFrameDir = "generatedFrames"

# which frames to generate captions between (frame1 = start, frame2 = end)
frame1 = 1
frame2 = 100

# to give viewer a bit more time to read the caption, add (up to) this many frames on both sides of the interval to display the caption
frameMargin = 8

# maximum amount of frame difference allowed for merging two captions
maxframediff = 7

# minimum and maximum area of caption allowed to display (as percentage of width*height)
minArea = 0.1
maxArea = 0.55

# how many captions to include in the final video
numCaptions = 200



# load data + font
myFont = ImageFont.truetype(fontPath, fontSize)
with open(jsonPath) as data_file:    
    data = json.load(data_file)

# get all captions
captions = getAllCaptions(frame1, frame2)

# remove oversized captions
captions = [c for c in captions if np.mean([r[2]*r[3] for r in c["rects"]]) < maxArea*wRaw*hRaw]
captions = [c for c in captions if np.mean([r[2]*r[3] for r in c["rects"]]) > minArea*wRaw*hRaw]

# now merging the frames. if you have a lot this may take a long time because I use
# a greedy, heuristic-based and inefficient way of merging them. it will merge in iterations and stop
# when there are no additional merges to make, although you can cut it off at any iteration
finished = False
nc = 0
while not finished:	
	captions = mergeCaptions(captions)
	if len(captions) == nc:
		finished = True
	nc = len(captions)
	print "num captions: %d" % nc
	

# normalize scores 
captions = normalizeCaptionScores(captions)

# sort captions by score
captions = sorted(captions, key=itemgetter("score"))
captions.reverse()


# draw the frames
for k in range(frame1,frame2):
	if k % 10 == 0:	print "render frame %d"%k
	im = drawFrame(k, captions, numCaptions, False)
	im.save(("%s/frame%04d.png"%(newFrameDir,k-frame1)), "png")


# create video
cmd ='ffmpeg -r 30 -i '+newFrameDir+'/frame%04d.png -r 30 -c:v libx264 -pix_fmt yuv420p out.mp4'
print cmd
os.system(cmd)

