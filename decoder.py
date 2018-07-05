from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys
import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)
        while state.buffer:
        # TODO: Write the body of this loop for part 4 
            vecc = self.extractor.get_input_representation(words, pos, state)
            possible = list(self.model.predict(vecc)[0])
            #informed by "https://stackoverflow.com/questions/4233476/sort-a-list-by-multiple-attributes"
            sorted_list = [j[0] for j in sorted(enumerate(possible), reverse=True, key=lambda x:x[1])]
            i=0
            t=self.output_labels[sorted_list[i]][0]
            while ((len(state.stack) == 0 and t in {"right_arc", "left_arc"}) 
                    or (len(state.stack) > 0 and len(state.buffer) == 1 and t == "shift") 
                    or (len(state.stack) > 0 and state.stack[-1] == 0 and t == "left_arc")):
                i+=1
                t=self.output_labels[sorted_list[i]][0]
            #retreives dependency structure
            if self.output_labels[sorted_list[i]][1] == None:
                state.shift()
            else:
                if self.output_labels[sorted_list[i]][0] == "left_arc":
                    state.left_arc(self.output_labels[sorted_list[i]][1])
                elif self.output_labels[sorted_list[i]][0] == "right_arc":
                    state.right_arc(self.output_labels[sorted_list[i]][1])
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
