import sys, os
import regex as re
import numpy as np

a = [
'Oi, Marcos. Bom dia.', 
'Oi, Marcos. Boa tarde.', 
'Oi, Marcos. Boa noite.',
'Oi, Marcos. Tudo bem?',
'Oi, Camila. Bom dia.', 
'Oi, Camila. Boa tarde.', 
'Oi, Camila. Boa noite.', 
'Oi, Camila. Tudo bem?',
'Oi, Luiza. Bom dia.', 
'Oi, Luiza. Boa tarde.', 
'Oi, Luiza. Boa noite.',
'Oi, Luiza. Tudo bem?'
]

def clean_str(s):
    s = re.sub("[^\P{P}]+", "", s)
    s = re.sub(" +", " ", s)
    return s.lower()


for s in a:
    print(s, len(s))
    
    sl = clean_str(s)
    print(sl, len(sl))
