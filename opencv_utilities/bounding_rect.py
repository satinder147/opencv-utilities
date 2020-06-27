import cv2
import numpy as np
import time


def check(scale, text, font, line_width, box_width, box_height, offset, p=1):
    """
    :param scale: parameter binary search is optimising
    :return: A boolean, whether this scale is ok or if p==2 sends back the string of words in each line.
    """
    last_word = 0
    prev_line_break = 0
    strings = []
    word_height = None
    for i in range(len(text)):
        if text[i] == ' ':
            last_word = i
        word_width, word_height = cv2.getTextSize(text[prev_line_break:i+1], font, scale, line_width)[0]
        if word_width > box_width:
            i = last_word
            # print("The text is "+text[prev_line_break:last_word],prev_line_break,last_word)
            if text[prev_line_break:last_word] == " " or last_word+1 == prev_line_break:
                return False
            strings.append(text[prev_line_break:last_word])
            prev_line_break = last_word+1
    strings.append(text[prev_line_break:len(text)])
    if p == 2:
        return strings 
    if (len(strings) * word_height + (len(strings) - 1) * offset) < box_height:
        return True
    else:
        return False


def get_scale(text, font, line_width, box_width, box_height, offset):
    lo = 0
    hi = 100
    while hi-lo > 1:
        mid = lo+(hi-lo)//2
        if check(mid, text, font, line_width, box_width, 
                 box_height, offset):
            lo = mid
        else:
            hi = mid
    increment = 0.1
    precision = 5
    ans = lo
    for _ in range(precision):
        while check(ans+increment, text, font, line_width, box_width, box_height, offset):
            ans += increment
        increment /= 10
    return ans


def get_image(x, y, box_width, box_height, image, text, pad_percent_height, pad_percent_width, line_width, align="left",
              font=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), rect=False):
    if rect:
        cv2.rectangle(image, (x, y), (x+box_width, y+box_height), (255, 0, 0), 1)
    padding = int(box_height*pad_percent_height)
    padding_width = int(pad_percent_width * box_width)
    box_width -= int(2*box_width*pad_percent_width)
    box_height -= int(2*box_height*pad_percent_height)
    offset = int(box_height/10)
    # print(box_width,box_height)
    ans = get_scale(text, font, line_width, box_width, box_height, offset)
    p = cv2.getTextSize(text, font, ans, line_width)
    x1 = x + padding_width
    y1 = y+p[0][1]+padding
    strings = check(ans, text, font, line_width, box_width, box_height, offset, p=2)

    for i in range(len(strings)):
        if align == 'left':
            cv2.putText(image, strings[i], (x1, y1), font, ans, color, line_width, cv2.LINE_AA)
        if align == 'center' or align == 'right':
            remaining = box_width-cv2.getTextSize(strings[i], font, ans, line_width)[0][0]
            if align == 'center':
                cv2.putText(image, strings[i], (x1+remaining//2, y1), font, ans, color, line_width,cv2.LINE_AA)
            else:
                cv2.putText(image, strings[i], (x1+remaining, y1), font, ans, color, line_width, cv2.LINE_AA)
        y1 += p[0][1]+offset
    return image


def get_transformed(text, pts_dest, img):

    pts_src = np.array([[0, 0], [600, 0], [600, 400], [0, 400]], dtype=float)
    src = np.zeros((400, 600, 3), dtype=np.uint8)
    src = get_image(0, 0, 600, 400, src, text, 0.05, 0.05, 3, 'center')
    h, status = cv2.findHomography(pts_src, pts_dest)
    im_temp = cv2.warpPerspective(src, h, (img.shape[1], img.shape[0]))
    grey = cv2.cvtColor(im_temp, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)[1]
    only_text = cv2.bitwise_and(im_temp, im_temp, mask=mask)
    only_back = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    result = cv2.bitwise_or(only_back, only_text)
    # cv2.imshow("result",result)
    # cv2.waitKey(0)
    return result


if __name__ == "__main__":
    """
    As per my knowledge there is no method in opencv that can automatically scale text
    inside a bounding box. My code uses binary search to find the optimal scale to fit the largest
    text inside the box. Apart from this it also provides functionality to align text and provide padding
    inside the box. It is as simple as just passing the coordinates of the box and the text.
    """
    st = time.time()
    text = '''I am Satinder Singh, engineering student from India.'''
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = get_image(10, 10, 100, 100, img, text, 0.05, 0.05, 1)
    cv2.imshow("results", result)
    cv2.waitKey(0)
    en = time.time()
    print(en-st)


