#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:47:32 2022

@author: juan
"""
import cv2
from cvzone.HandTrackingModule import HandDetector

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon = 0.8, maxHands = 2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw= True)
    
    if hands:
        #for hand 1
        hand1 = hands[0]
        lmList1 = hand1['lmList'] #List of 21 landmarks points
        bbox1 = hand1['bbox'] #Bounding box info -> x,y,w,h
        centerPoint1 = hand1['center'] #center of the hand cx, cy
        handType1 = hand1['type'] #Hand type left or right
        
        fingers1 = detector.fingersUp(hand1)
        
        print(fingers1)
        
        if len(hands) == 2:
            #for hand 2
            hand2 = hands[1]
            lmList2 = hand2['lmList'] #List of 21 landmarks points
            bbox2 = hand2['bbox'] #Bounding box info -> x,y,w,h
            centerPoint2 = hand2['center'] #center of the hand cx, cy
            handType2 = hand2['type'] #Hand type left or right
            
            fingers2 = detector.fingersUp(hand2)
            print(fingers1, fingers2)
            
            if fingers2 == [1,0,1,0,0]:
                cv2.putText(img, 'FCKU', (10,500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        
        if fingers1 == [1,0,1,0,0]:
            cv2.putText(img, 'FCKU', (10,500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
            
        

    cv2.imshow('Image', img)
    cv2.waitKey(1)

