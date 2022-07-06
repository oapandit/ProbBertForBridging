import os
from bridging_utils import *
from bridging_json_extract import BridgingJsonDocInformationExtractor
from transformers import BertTokenizer, BertModel, BertConfig, BertLMHeadModel
import torch
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import collections
from scipy.stats import entropy

head_key = "improvedHead"

lm_model = "bert-base-cased"
# lm_model = "bert-large-cased"
# lm_model = "roberta-base"
# lm_model = "roberta-large"

class AnalyzeBERT:
    def __init__(self, json_objects_path,jsonlines,is_head_sent):
        self.json_objects_path = json_objects_path
        self.jsonlines = jsonlines
        self.is_head_sent = is_head_sent


    def replace_word(self,word):
        word = word.strip()
        if word.lower() == "inc." or word.lower() == "inc":
            word = "incorporation"
        if word.lower() == "co." or word.lower() == "co":
            word = "company"
        if word.lower() == "%":
            word = "percent"
        if word.lower() == "$":
            word = "dollar"
        if word.lower() == "anc":
            word = "congress"
        if word.lower() == "corp." or word.lower() == "corp":
            word = "corporation"
        return word

    def find_all_mention_start_end_indicators(self,target_lists):
        start_indicators = []
        start_sublist = ['$', 's', '##s']
        end_sublist = ['$', 'es']

        start_sublist_len = len(start_sublist)
        end_sublist_len = len(end_sublist)
        final_results = []
        for target_list in target_lists:
            logger.debug("-----")
            logger.debug("{}".format(target_list))
            results = []
            for ind in (i for i, e in enumerate(target_list) if e == start_sublist[0]):
                if target_list[ind:ind + start_sublist_len] == start_sublist:
                    start_ind = (ind, ind + start_sublist_len - 1)
                    start_indicators.append(start_ind)
                elif target_list[ind:ind + end_sublist_len] == end_sublist:
                    end_ind = (ind, ind + end_sublist_len - 1)
                    results.append((start_indicators.pop(), end_ind))
            results.sort()
            logger.debug("markers at {}".format(results))
            final_results.append(results)
        return final_results

    def get_json_objects(self, jsonlines):
        json_path = os.path.join(self.json_objects_path, jsonlines)
        json_objects = read_jsonlines_file(json_path)
        return json_objects


    def get_sentences_marked(self,bridging_json):
        ana_ante_indices = bridging_json.anaphors + bridging_json.antecedents
        _, head_spans = bridging_json.get_spans_head(
            ana_ante_indices, True)

        ana_ante_sents = bridging_json.get_span_sentences(ana_ante_indices)
        ana_ante_head_sents = bridging_json.get_span_sentences(head_spans)

        return ana_ante_indices,ana_ante_sents,ana_ante_head_sents

    def create_sentences_containing_ana_ante(self,jsonlines):
        json_objects = self.get_json_objects(jsonlines)
        logger.debug("********** generating vecs for json docs ***********")
        bert_ana_ante_sents = []
        for single_json_object in json_objects:
            bridging_json = BridgingJsonDocInformationExtractor(single_json_object, logger)
            if bridging_json.is_file_empty:
                print("doc {} is empty".format(bridging_json.doc_key))
                continue
            anaphors, antecedents =bridging_json.anaphors, bridging_json.antecedents
            if self.is_head_sent:
                #if consiring heads of mentions; get the anaphor antecedent head indices
                _, anaphors = bridging_json.get_spans_head(bridging_json.anaphors, True)
                _, antecedents = bridging_json.get_spans_head(bridging_json.antecedents, True)

            for (ana,ante) in zip(anaphors, antecedents):
                curr_ana_ante_sent = bridging_json.get_between_sentences_marked(ante,ana,is_get_between_sents=False)
                bert_ana_ante_sents.append(curr_ana_ante_sent)

                ana_sent_num = bridging_json.get_sent_num(ana)
                ante_sent_num = bridging_json.get_sent_num(ante)
                if ana_sent_num == ante_sent_num:
                    print(curr_ana_ante_sent)

        print(bert_ana_ante_sents[0:5])
        assert len(bert_ana_ante_sents) == 663,"{} number of sentences".format(len(bert_ana_ante_sents))
        return bert_ana_ante_sents

    def create_sentences_from_ante_to_ana(self,jsonlines):
        sent_distance_to_ana_ante_sentences = {}
        json_objects = self.get_json_objects(jsonlines)
        logger.debug("********** generating vecs for json docs ***********")

        for single_json_object in json_objects:
            bridging_json = BridgingJsonDocInformationExtractor(single_json_object, logger)
            if bridging_json.is_file_empty:
                print("doc {} is empty".format(bridging_json.doc_key))
                continue
            anaphors, antecedents =bridging_json.anaphors, bridging_json.antecedents
            # if self.is_head_sent:
            #     if consiring heads of mentions; get the anaphor antecedent head indices
            #     _, anaphors = bridging_json.get_spans_head(bridging_json.anaphors, True)
            #     _, antecedents = bridging_json.get_spans_head(bridging_json.antecedents, True)

            ana_to_antecedents_map = bridging_json.get_anaphor_to_antecedents_map(bridging_json.bridging_clusters + bridging_json.art_bridging_pairs_good)
            for ana in anaphors:
                ana_sent_num = bridging_json.get_sent_num(ana)
                if self.is_head_sent:
                    _,ana_head = bridging_json.get_span_head(ana,is_semantic_head=True)
                for ante in ana_to_antecedents_map[ana]:
                    ante_sent_num = bridging_json.get_sent_num(ante)

                    assert ante_sent_num <= ana_sent_num,"ante sent : {} and ana sent : {}".format(ante_sent_num,ana_sent_num)
                    distance = ana_sent_num - ante_sent_num
                    assert distance >=0

                    if self.is_head_sent:
                        _, ante_head = bridging_json.get_span_head(ante, is_semantic_head=True)
                        curr_ana_ante_sents = bridging_json.get_between_sentences_marked(ante_head, ana_head)
                    else:
                        curr_ana_ante_sents = bridging_json.get_between_sentences_marked(ante,ana)
                    curr_dist_list = sent_distance_to_ana_ante_sentences.get(distance,[])
                    curr_dist_list.append(curr_ana_ante_sents)
                    sent_distance_to_ana_ante_sentences[distance] = curr_dist_list

        total_ana = 0
        for k in sorted(sent_distance_to_ana_ante_sentences.keys()):
            num_ana_k = len(sent_distance_to_ana_ante_sentences[k])
            print(5 * "======")
            print("distance {} and number of anaphors {}".format(k,num_ana_k))
            # for s in sent_distance_to_ana_ante_sentences[k][0:5]:
            #     print(s)
            #     print(5*"---")
            total_ana += num_ana_k
        assert total_ana >= 663,total_ana
        return sent_distance_to_ana_ante_sentences


    def get_bert_attn_weights(self,bert_ana_ante_sents):
        config = BertConfig.from_pretrained("bert-base-cased", output_attentions=True, num_labels=2)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = BertModel.from_pretrained('bert-base-cased', config=config)

        inputs = tokenizer(bert_ana_ante_sents, return_tensors="pt", padding=True)
        outputs = model(**inputs)

        input_ids = inputs["input_ids"].tolist()
        text_tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]

        return outputs[-1],text_tokens,input_ids

    def get_attn_weights_for_ana_ante(self,complete_sents_attn_weights,text_tokens,input_ids):
        marker_indices = self.find_all_mention_start_end_indicators(text_tokens)
        ana_ant_attn_weights_per_layer = np.zeros((12, 12)).tolist()
        ant_ana_attn_weights_per_layer = np.zeros((12, 12)).tolist()
        ana_ant_attn_weight_dist = {"l": [0, 0], "m": [0, 0], "h": [0, 0]}
        ant_ana_attn_weight_dist = {"l": [0, 0], "m": [0, 0], "h": [0, 0]}
        # lay = {"l":[0,1,2],"m":[5,6,7],"h":[9,10,11]}
        l_lay= [0, 1, 2]
        m_lay=  [5, 6, 7]
        h_lay=  [9, 10, 11]
        num_pairs = 0
        for i, marker_index in enumerate(marker_indices):
            num_pairs += 1
            ant_s = marker_index[0][0][-1] + 1
            ant_e = marker_index[0][1][0]

            ana_s = marker_index[1][0][-1] + 1
            ana_e = marker_index[1][1][0]

            sent = text_tokens[i]
            # print("ana : {}, ante : {}".format(sent[ana_s:ana_e], sent[ant_s:ant_e]))

            for layer in range(12):
                if layer in l_lay:
                    _lay = "l"
                elif layer in m_lay:
                    _lay = "m"
                elif layer in h_lay:
                    _lay = "h"
                else:
                    _lay = "n"
                for attn in range(12):
                    s = []
                    for ana_ind in range(ana_s, ana_e + 1):
                        # print("layer : {}, i : {},attn :{}, ana ind : {}, ant : {} - {}".format(layer,i,attn,ana_ind,ant_s,ant_e+1))
                        _weight = sum(complete_sents_attn_weights[layer][i][attn][ana_ind][ant_s:ant_e + 1].tolist())
                        cum_weight = sum(complete_sents_attn_weights[layer][i][attn][ana_ind][1:input_ids[i].index(102)].tolist())
                        assert _weight < cum_weight
                        attn_weight_mass = _weight/cum_weight
                        s.append(attn_weight_mass)

                        if _lay != "n":
                            ana_ant_attn_weight_dist[_lay][0] += _weight
                            ana_ant_attn_weight_dist[_lay][1] += cum_weight

                    ana_ant_attn_weights_per_layer[layer][attn] += average(s)



                    s = []
                    for ante_ind in range(ant_s, ant_e + 1):
                        # print("layer : {}, i : {},attn :{}, ana ind : {}, ant : {} - {}".format(layer,i,attn,ante_ind,ana_s,ana_e+1))
                        _weight = sum(complete_sents_attn_weights[layer][i][attn][ante_ind][ana_s:ana_e + 1].tolist())
                        cum_weight = sum(complete_sents_attn_weights[layer][i][attn][ante_ind][1:input_ids[i].index(102)].tolist())
                        attn_weight_mass = _weight/cum_weight
                        assert _weight < cum_weight
                        s.append(attn_weight_mass)

                        if _lay != "n":
                            ant_ana_attn_weight_dist[_lay][0] += _weight
                            ant_ana_attn_weight_dist[_lay][1] += cum_weight

                    ant_ana_attn_weights_per_layer[layer][attn] += average(s)

            if i%25==0:
                print("processed {} sentences".format(i))

        ana_ant_attn_weights_per_layer = np.array(ana_ant_attn_weights_per_layer)/num_pairs
        ant_ana_attn_weights_per_layer = np.array(ant_ana_attn_weights_per_layer)/num_pairs
        return ana_ant_attn_weights_per_layer,ant_ana_attn_weights_per_layer,ana_ant_attn_weight_dist,ant_ana_attn_weight_dist

    def get_attn_weights_for_ana_ante_for_prom_heads(self,complete_sents_attn_weights,text_tokens,input_ids):
        marker_indices = self.find_all_mention_start_end_indicators(text_tokens)
        ana_ant_attn_weight_dist_prom_heads = [0, 0]
        ant_ana_attn_weight_dist_prom_heads = [0, 0]

        prom_heads= [[4,0],[8,11],[10,2]]
        num_pairs = 0

        easy_pairs = []
        difficult_pairs = []
        for i, marker_index in enumerate(marker_indices):
            num_pairs += 1
            ant_s = marker_index[0][0][-1] + 1
            ant_e = marker_index[0][1][0]

            ana_s = marker_index[1][0][-1] + 1
            ana_e = marker_index[1][1][0]

            sent = text_tokens[i]
            # print("ana : {}, ante : {}".format(sent[ana_s:ana_e], sent[ant_s:ant_e]))
            ana_wt, ana_cwt = 0, 0
            at_wt, at_cwt = 0, 0
            for l_h in prom_heads:
                layer,attn = l_h


                for ana_ind in range(ana_s, ana_e + 1):
                    # print("layer : {}, i : {},attn :{}, ana ind : {}, ant : {} - {}".format(layer,i,attn,ana_ind,ant_s,ant_e+1))
                    _weight = sum(complete_sents_attn_weights[layer][i][attn][ana_ind][ant_s:ant_e + 1].tolist())
                    _cum_weight = sum(
                        complete_sents_attn_weights[layer][i][attn][ana_ind][1:input_ids[i].index(102)].tolist())

                    ana_ant_attn_weight_dist_prom_heads[0] += _weight
                    ana_ant_attn_weight_dist_prom_heads[1] += _cum_weight

                    ana_wt += _weight
                    ana_cwt += _cum_weight

                for ante_ind in range(ant_s, ant_e + 1):
                    # print("layer : {}, i : {},attn :{}, ana ind : {}, ant : {} - {}".format(layer,i,attn,ante_ind,ana_s,ana_e+1))
                    _weight = sum(complete_sents_attn_weights[layer][i][attn][ante_ind][ana_s:ana_e + 1].tolist())
                    cum_weight = sum(
                        complete_sents_attn_weights[layer][i][attn][ante_ind][1:input_ids[i].index(102)].tolist())

                    ant_ana_attn_weight_dist_prom_heads[0] += _weight
                    ant_ana_attn_weight_dist_prom_heads[1] += cum_weight

                    at_wt += _weight
                    at_cwt += cum_weight

            perc_attn = ana_wt * 100.0 / ana_cwt
            if perc_attn > 70:
                easy_pairs.append(i)
            if perc_attn < 10:
                difficult_pairs.append(i)

            if i%25==0:
                print("processed {} sentences".format(i))
        return ana_ant_attn_weight_dist_prom_heads,ant_ana_attn_weight_dist_prom_heads,easy_pairs,difficult_pairs

    def plot_heatmap(self,ana_ant_attn_weights_per_layer,ant_ana_attn_weights_per_layer,plot_name):
        plot_f_path = os.path.join(bert_analysis_path,plot_name)
        nums = list(range(1, 13))
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 13))
        # im = ax.imshow(harvest)
        # ax = sns.heatmap(ana_ant_attn_weights_per_layer, linewidth=0.5)
        ax.imshow(ana_ant_attn_weights_per_layer, cmap='magma_r', interpolation='nearest',origin='lower')
        ax.set_xticks(np.arange(len(nums)))
        ax.set_yticks(np.arange(len(nums)))
        ax.set_xticklabels(nums)
        ax.set_yticklabels(nums)
        # ax.set_title("Anaphor to Antecedent attention weights")

        # grid.cbar_axes[1].colorbar(im1)
        # ax.colorbar()

        ax2.imshow(ant_ana_attn_weights_per_layer, cmap='magma_r', interpolation='nearest',origin='lower')
        ax2.set_xticks(np.arange(len(nums)))
        ax2.set_yticks(np.arange(len(nums)))
        ax2.set_xticklabels(nums)
        ax2.set_yticklabels(nums)
        # ax2.set_title("Antecedent to Anaphor attention weights")

        # ax.set_xlabel('Attention Heads')
        # ax.set_ylabel('Layers')

        # ax2.set_xlabel('Attention Heads')
        # ax2.set_ylabel('Layers')

        fig.tight_layout(pad=3.0)
        # fig.show()
        fig.savefig(plot_f_path, bbox_inches="tight")

        print("ana to ante")
        for i in range(ana_ant_attn_weights_per_layer.shape[0]):
            print(ana_ant_attn_weights_per_layer[i])

        print("ante to ana")
        for i in range(ana_ant_attn_weights_per_layer.shape[0]):
            print(ant_ana_attn_weights_per_layer[i])


    def plot_bar_percentage_attn(self,attn_weight_dist_distances, plot_name):
        def get_wt_percent(wt_dist,lay):
            assert wt_dist[lay][0] < wt_dist[lay][1],"wt {}, cum wt {}".format(wt_dist[lay][0],wt_dist[lay][1])
            return wt_dist[lay][0] * 100 / wt_dist[lay][1]
        lower_layers = []
        middle_layers = []
        higher_layers = []
        for attn_weight_dist in attn_weight_dist_distances:
            lower_layers.append(get_wt_percent(attn_weight_dist,"l"))
            middle_layers.append(get_wt_percent(attn_weight_dist, "m"))
            higher_layers.append(get_wt_percent(attn_weight_dist, "h"))
        assert len(lower_layers) == len(middle_layers) == len(higher_layers)
        plot_name = plot_name if not self.is_head_sent else "{}_head".format(plot_name)
        plot_f_path = os.path.join(bert_analysis_path, "{}_attn_weights_per_layer.png".format(plot_name))
        fig, ax = plt.subplots()
        # set width of bar
        barWidth = 0.25

        # Set position of bar on X axis
        r1 = np.arange(len(lower_layers))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]

        # Make the plot
        ax.bar(r1, lower_layers, color='lightblue', width=barWidth, edgecolor='white', label='var1')
        ax.bar(r2, middle_layers, color='lightgreen', width=barWidth, edgecolor='white', label='var2')
        ax.bar(r3, higher_layers, color='lightcoral', width=barWidth, edgecolor='white', label='var3')

        # Add xticks on the middle of the group bars
        ax.set_xlabel('Distance between anaphor-antecedent', fontweight='bold')
        ax.set_ylabel('Percentage of attention', fontweight='bold')
        ax.set_xticks([r + barWidth for r in range(len(lower_layers))])
        ax.set_xticklabels(['0', '1', '2', '3-5', '6-10'])
        # Create legend & Show graphic
        ax.legend(('lower', 'middle', 'higher'))
        # plt.show()
        fig.savefig(plot_f_path, bbox_inches="tight")

    def plot_bar_percentage_attn_prominent_heads(self,attn_weight_dist_distances, plot_name):
        bars = []
        for attn_weight_dist in attn_weight_dist_distances:
            perc = attn_weight_dist[0] * 100 / attn_weight_dist[1]
            bars.append(perc)

        plot_name = plot_name if not self.is_head_sent else "{}_head".format(plot_name)
        plot_f_path = os.path.join(bert_analysis_path, "{}_prominent_attn_heads.png".format(plot_name))

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        dists = ['0', '1', '2', '3-5']

        ax.bar(dists, bars)


        # Add xticks on the middle of the group bars
        ax.set_xlabel('Distance between anaphor-antecedent', fontweight='bold')
        ax.set_ylabel('Percentage of attention', fontweight='bold')
        # plt.show()
        fig.savefig(plot_f_path, bbox_inches="tight")

    def analyze_ana_ante_attn_weights_all_dist_sents(self):
        # bert_ana_ante_sents = self.create_sentences_containing_ana_ante(self.jsonlines)
        sent_distance_to_ana_ante_sentences = self.create_sentences_from_ante_to_ana(self.jsonlines)
        distances = [0,1,2,[3,5],[6,10]] # we are plotting different heatmaps for different distances.
        distance_names = [0, 1, 2, "3_4_5", "6_7_8_9_10"]
        ana_ant_attn_weight_dist_distances, ant_ana_attn_weight_dist_distances = [],[]
        for i,dist in enumerate(distances):
            if isinstance(dist,list):
                bert_ana_ante_sents = []
                d = dist[0]
                d_end = dist[-1] if d!=11 else 36
                while d<=d_end:
                    if sent_distance_to_ana_ante_sentences.get(d,None) is not None:
                        bert_ana_ante_sents += sent_distance_to_ana_ante_sentences[d]
                    d+=1
            else:
                bert_ana_ante_sents = sent_distance_to_ana_ante_sentences[dist]
            print(bert_ana_ante_sents[-5:-1])
            complete_sents_attn_weights, text_tokens,input_ids = self.get_bert_attn_weights(bert_ana_ante_sents)
            ana_ant_attn_weights_per_layer, ant_ana_attn_weights_per_layer,ana_ant_attn_weight_dist,ant_ana_attn_weight_dist = self.get_attn_weights_for_ana_ante(complete_sents_attn_weights, text_tokens,input_ids)
            ana_ant_attn_weight_dist_distances.append(ana_ant_attn_weight_dist)
            ant_ana_attn_weight_dist_distances.append(ant_ana_attn_weight_dist)
            plot_name = "heatmap_distance_{}".format(distance_names[i])
            plot_name = plot_name if not self.is_head_sent else "{}_head".format(plot_name)
            self.plot_heatmap(ana_ant_attn_weights_per_layer, ant_ana_attn_weights_per_layer,plot_name+".png")

        # self.plot_bar_percentage_attn(ana_ant_attn_weight_dist_distances,"ana_ante")
        # self.plot_bar_percentage_attn(ant_ana_attn_weight_dist_distances, "ante_ana")

    def _log_sentences(self,sentences,indices):
        for i in indices:
            logger.critical("i : {}".format(i))
            logger.critical(sentences[i])
            logger.critical(3*"----")

    def analyze_ana_ante_attn_weights_all_dist_sents_prominent_heads(self):
        # bert_ana_ante_sents = self.create_sentences_containing_ana_ante(self.jsonlines)
        sent_distance_to_ana_ante_sentences = self.create_sentences_from_ante_to_ana(self.jsonlines)
        distances = [0,1,2,[3,5]] # we are plotting different heatmaps for different distances.
        distance_names = [0, 1, 2, "3_4_5", "6_7_8_9_10"]
        ana_ant_attn_weight_dist_distances, ant_ana_attn_weight_dist_distances = [],[]
        for i,dist in enumerate(distances):
            if isinstance(dist,list):
                bert_ana_ante_sents = []
                d = dist[0]
                d_end = dist[-1] if d!=11 else 36
                while d<=d_end:
                    if sent_distance_to_ana_ante_sentences.get(d,None) is not None:
                        bert_ana_ante_sents += sent_distance_to_ana_ante_sentences[d]
                    d+=1
            else:
                bert_ana_ante_sents = sent_distance_to_ana_ante_sentences[dist]
            print(bert_ana_ante_sents[-5:-1])
            complete_sents_attn_weights, text_tokens,input_ids = self.get_bert_attn_weights(bert_ana_ante_sents)
            ana_ant_attn_weight_dist,ant_ana_attn_weight_dist,easy_pairs,difficult_pairs = self.get_attn_weights_for_ana_ante_for_prom_heads(complete_sents_attn_weights, text_tokens,input_ids)
            logger.critical(3*"=====++++=====")
            logger.critical("sentences which are difficult to recognize from anaphor to antecedents at distance {}".format(dist))
            self._log_sentences(bert_ana_ante_sents,difficult_pairs)
            logger.critical(3 * "=====++++=====")
            logger.critical(
                "sentences which are easy to recognize from anaphor to antecedents at distance {}".format(dist))
            self._log_sentences(bert_ana_ante_sents, easy_pairs)

            # ana_ant_attn_weight_dist_distances.append(ana_ant_attn_weight_dist)
            # ant_ana_attn_weight_dist_distances.append(ant_ana_attn_weight_dist)

        # self.plot_bar_percentage_attn_prominent_heads(ana_ant_attn_weight_dist_distances,"ana_ante")
        # self.plot_bar_percentage_attn_prominent_heads(ant_ana_attn_weight_dist_distances, "ante_ana")

    def analyze_ana_ante_attn_weights(self):
        bert_ana_ante_sents = self.create_sentences_containing_ana_ante(self.jsonlines)
        complete_sents_attn_weights, text_tokens,input_ids = self.get_bert_attn_weights(bert_ana_ante_sents)
        ana_ant_attn_weights_per_layer, ant_ana_attn_weights_per_layer,ana_ant_attn_weight_dist,ant_ana_attn_weight_dist = self.get_attn_weights_for_ana_ante(complete_sents_attn_weights, text_tokens,input_ids)
        plot_name = "heatmap_only_ana_ant_sent"
        plot_name = plot_name if not self.is_head_sent else "{}_head".format(plot_name)
        self.plot_heatmap(ana_ant_attn_weights_per_layer, ant_ana_attn_weights_per_layer,plot_name+".png")


    def read_prob_dataset(self,prob_path):
        json_obj = read_jsonlines_file(prob_path)[0]
        for c,i in enumerate(json_obj):
            if c%20==0 and c==0:
                print(c)
                print(i["OrigContext"])
                print("\n \n")
                # print(i["newContext"])
                # print("\n \n")
                # print(i["newContext"].replace('[MASK]', i["antecedents"][0]['head']))
                # print("\n \n")
                print("Anaphor : ", i["anaphor"])
                print("\n \n")
                print("Antecedents : ", i["antecedents"])
                print("\n \n")
                print("Options : ", i["options"])
                print(5 * "----")
                print("\n \n \n")
                break
        print("number of ana {}".format(len(json_obj)))
        # sys.exit()
        return json_obj


    def get_common_anaphors(self,json_data_all_sents_obj,json_data_2sents_obj):
        common_jason_obj = []
        un_common_jason_obj = []

        #create dictionary
        ana_2sent_dict = {sentence["anaphor"]+sentence["OrigContext"]:counter for counter,sentence in enumerate(json_data_2sents_obj)}
        num_ana = 0
        total = 0
        for sentence in json_data_all_sents_obj:
            curr_ana = sentence["anaphor"]
            orig_context = sentence["OrigContext"]
            if ana_2sent_dict.get(curr_ana+orig_context,None) is not None:
                num_ana += 1
                # sentence_loc_2sent = ana_2sent_dict[curr_ana]
                # orig_context = sentence["OrigContext"]
                # orig_context_2sent = json_data_2sents_obj[sentence_loc_2sent]["OrigContext"]
                # assert orig_context == orig_context_2sent, "[{}] is not same as [{}]".format(orig_context, orig_context_2sent)
                common_jason_obj.append(sentence)
            else:
                un_common_jason_obj.append(sentence)
            total += 1
        print("total anaphors with all sentence : {} have 2 sent candidates : {}  and those do not have {}".format(total,num_ana,len(un_common_jason_obj)))
        return common_jason_obj,un_common_jason_obj


    def get_sentence_prob_accuracy_bert_compliant_sentences(self, json_obj):
        total = 0
        correct = 0
        model = BertLMHeadModel.from_pretrained('bert-base-cased',is_decoder=True)
        model.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        def score(sentence):
            if len(sentence.strip().split()) <= 1: return 10000
            tokenize_input = tokenizer.tokenize(sentence)
            if len(tokenize_input) > 512: return 10000
            input_ids = torch.tensor(tokenizer.encode(tokenize_input)).unsqueeze(0)
            with torch.no_grad():
                loss = model(input_ids, labels=input_ids)[0]
            return math.exp(loss.item() / len(tokenize_input))

        for counter,sentence in enumerate(json_obj):
            new_context = sentence["newContext"].replace('[MASK]', '{}')
            options = sentence["options"]
            opt_scores = []
            # if counter % 10 == 0:
            print(3*"====+++===")
            print("opts : {}".format(options))
            print("anaphor {}".format(sentence["anaphor"]))
            print("antecedent {}".format(sentence["antecedent"]))
            print("new context : {}".format(new_context))
            for c,option in enumerate(options):
                print(c)
                sent = sentence["newContext"].replace('[MASK]', option)
                sc = score(sent)
                opt_scores.append(sc)
                # if counter %10 == 0:
                print("sent : {}".format(sent))
                print("opt : {}".format(option))
                print("score : {}".format(sc))
                print(5*"---")

            if opt_scores.index(min(opt_scores)) == 0:
                # if counter % 10 == 0:
                print("accurate pred")
                correct += 1
            total +=1
        print("total : {} , correct : {} and accuracy : {}".format(total,correct,correct*100.0/total))


    def _get_target_acc(self,counter,new_context,antecedents,options,nlp,ante_heads,num_cand_ant, num_score_null, num_head_abs, num_error, correct):

        logger.debug(5 * "--")
        logger.debug("sent given to model: {}".format(new_context))
        logger.debug("antecedents : {}".format(antecedents))
        # logger.debug("antecedents heads : {}".format(ante_heads))

        is_correct = False
        opt_heads = []
        scores = []
        for c, option_dict in enumerate(options):
            num_cand_ant += 1
            opt_heads.append(option_dict[head_key])
            target = ' {}'.format(self.replace_word(option_dict[head_key]))
            # print("sent : {}".format(new_context))
            # print("opt : {}".format(option_dict))
            # print("target : {}".format(target))

            _score = nlp(new_context, targets=[target])[0]['score']
            scores.append(_score)
            if counter % 30 == 0:
                logger.debug("opt : {}".format(option_dict))
                logger.debug("target : {}".format(target))
                logger.debug("score : {}".format(_score))
                logger.debug(2 * "---")

        assert len(scores) == len(options)
        try:
            assert len(scores) > 0
            max_ind = np.asarray(scores).argmax()
            sel_head = opt_heads[max_ind]

            for ante_head in ante_heads:
                if ante_head in opt_heads:
                    is_head_present = True
                    break
            assert is_head_present
        except:
            num_error += 1
            logger.critical(5 * "**")
            if len(scores) == 0:
                num_score_null += 1
                logger.critical("ERROR : SCORES ZERO")
            elif not is_head_present:
                num_head_abs += 1
                logger.critical("ERROR : HEAD NOT PRESENT IN OPTIONS")
            else:
                logger.critical("ERROR : UNKNOWN")
            logger.critical("sent : {}".format(new_context))
            logger.critical("heads : {} ".format(ante_heads))
            logger.critical("options : {}".format(opt_heads))
            logger.critical("scores {}".format(scores))

        if sel_head in ante_heads:
            correct += 1
            is_correct = True
        if counter % 30 == 0:
            logger.debug("ante heads : {}".format(ante_heads))
            logger.debug("selected head : {}".format(sel_head))
            logger.debug("is correct : {}".format(is_correct))
        logger.debug(5 * "--")
        return is_correct,scores,sel_head,num_cand_ant, num_score_null, num_head_abs, num_error, correct

    def get_target_prob_accuracy_bert_compliant_sentences(self, json_obj,target_json_path):

        from transformers import pipeline

        total = 0
        correct = 0
        nlp = pipeline("fill-mask", model=lm_model)
        num_cand_ant = 0
        num_ant_two_masks = 0
        num_error = 0
        num_score_null, num_head_abs = 0, 0
        pred_jsons = []
        for counter,sentence in enumerate(json_obj):
            is_correct = False
            # new_context = sentence["newContext"].replace('[MASK]', '{nlp.tokenizer.mask_token}')
            new_context = sentence["newContext"]
            if new_context.count("of [MASK]") > 1:
                num_ant_two_masks +=1
                instances = new_context.count("of [MASK]") - 1
                new_context = new_context.replace("of [MASK] ", "", instances)
            if "roberta" in lm_model:
                new_context = new_context.replace("[MASK]","<mask>")
            options = sentence["options"]
            antecedents = sentence["antecedents"]
            ante_heads = [d[head_key] for d in antecedents]
            # print("heads : {} ".format(ante_heads))
            scores = []
            opt_heads = []
            try:
                is_correct,scores,sel_head,num_cand_ant, num_score_null, num_head_abs, num_error, correct = self._get_target_acc(counter, new_context, antecedents, options, nlp, ante_heads,num_cand_ant, num_score_null, num_head_abs, num_error, correct)

            finally :
                total += 1
                sentence['pred_prob'] = scores
                sys_entro = entropy(scores, base=2)
                sentence['entropy'] = sys_entro
                sentence['is_correct'] = is_correct
                sentence['selected_ante'] = sel_head
                sentence["newContext"] = new_context
                pred_jsons.append(sentence)
            if counter % 20 == 0 and counter !=0:
                print("remaining {}".format(len(json_obj)-counter))
                # break


        print("total : {} , correct : {} and accuracy : {}".format(total,correct,correct*100.0/total))
        print("accuracy from total anaphors {}".format(correct*100.0/663))
        print("total erros : {}, num two masks :{}, num of null scores : {} and number of heads absent : {}".format(num_error,
                                                                                                  num_ant_two_masks,
                                                                                                  num_score_null,
                                                                                                  num_head_abs))
        print("total candidate antecedents : {} and average {}".format(num_cand_ant,num_cand_ant/total))

        logger.critical(5*"===")
        logger.critical("final result with model {}".format(lm_model))
        logger.critical(5 * "===")
        logger.critical("total : {} , correct : {} and accuracy : {}".format(total,correct,correct*100.0/total))
        logger.critical("accuracy from total anaphors {}".format(correct*100.0/663))
        logger.critical("total erros : {}, num two masks :{}, num of null scores : {} and number of heads absent : {}".format(num_error,                                                                                         num_ant_two_masks,
                                                                                                  num_score_null,
                                                                                                  num_head_abs))
        logger.critical("total candidate antecedents : {} and average {}".format(num_cand_ant,num_cand_ant/total))
        target_json_path += "_"+lm_model
        write_jsonlines_file(pred_jsons,target_json_path)
        self.analyze_preds_with_entropy(pred_jsons)
        return pred_jsons

    def analyze_preds_with_entropy(self,pred_jsons=None,pred_jsons_path = None):
        if pred_jsons is None:
            assert pred_jsons_path is not None
            pred_jsons = read_jsonlines_file(pred_jsons_path)

        def write_in_log(is_log_correct):
            logger.critical(5 * "+++++")
            logger.critical(5 * "ANALYZE {}".format("CORRECT" if is_log_correct else "INCORRECT"))
            logger.critical(5 * "+++++")

            # keys = ordered_entropy_to_sentence_counter.keys()
            keys = a1_sorted_keys
            if is_log_correct:
                keys.reverse()
            counter = 0
            for i, key in enumerate(keys):
                sentence = pred_jsons[entropy_to_sentence_counter[key]]

                orig_context = sentence["newContext"]
                anaphor = sentence["anaphor"]

                options = sentence["options"]
                antecedents = sentence["antecedents"]
                ante_heads = [d[head_key] for d in antecedents]
                antecedents = [ant['mentionStr'] for ant in antecedents]

                opt_heads = []
                for c,option_dict in enumerate(options):
                    opt_heads.append(option_dict[head_key])

                sel_head = sentence['selected_ante']
                is_correct = sentence['is_correct']
                if is_correct == is_log_correct:
                    logger.critical(5 * "==++==")
                    logger.critical("entropy : {}".format(sentence['entropy']))
                    logger.critical("sent : {}".format(orig_context))
                    logger.critical("anaphor : {} - {}".format(anaphor['mentionStr'],anaphor['head']))
                    logger.critical("antecedents : {}".format(antecedents))
                    logger.critical("antecedents heads : {}".format(ante_heads))

                    logger.critical("Available options {}".format(opt_heads))
                    logger.critical("selected head : {}".format(sel_head))
                    logger.critical("is correct : {}".format(is_correct))

                    counter += 1

                if counter == 100:
                    break

        entropy_to_sentence_counter = {}
        for counter, sentence in enumerate(pred_jsons):
            # print(sentence)
            ent = sentence['entropy']
            assert entropy_to_sentence_counter.get(ent,None) is None
            entropy_to_sentence_counter[ent] = counter

        # ordered_entropy_to_sentence_counter = collections.OrderedDict(sorted(entropy_to_sentence_counter.items()))
        a1_sorted_keys = sorted(entropy_to_sentence_counter, key=entropy_to_sentence_counter.get)
        write_in_log(is_log_correct=True)

        write_in_log(is_log_correct=False)


    def _mask_adj(self,new_context,num_ant_two_masks,_mask):
        if new_context.count(_mask) > 1:
            num_ant_two_masks += 1
            instances = new_context.count(_mask) - 1
            new_context = new_context.replace("{} ".format(_mask), "", instances)
        if "roberta" in lm_model:
            new_context = new_context.replace("[MASK]", "<mask>")
        return new_context,num_ant_two_masks

    def get_target_prob_acc_depending_context(self,json_obj,target_json_path,is_without_of=False,is_with_pert=False):
        from transformers import pipeline

        _mask = "[MASK]" if is_without_of else "of [MASK]"
        total = 0
        total_ante_sent_added = 0
        correct_with_context = 0
        correct_without_any_context = 0
        correct_only_with_anaphor_sent = 0
        correct_with_ante_sent_added = 0

        nlp = pipeline("fill-mask", model=lm_model)
        num_cand_ant = 0
        num_ant_two_masks = 0
        num_error = 0
        num_score_null, num_head_abs = 0, 0
        pred_jsons = []
        for counter,sentence in enumerate(json_obj):
            # if counter == 10:
            #     break
            logger.debug(5 * "*******")
            logger.debug("counter : {}".format(counter))
            logger.debug("*******")
            is_with_context_correct = False
            is_correct_only_with_anaphor_sent = False
            is_correct_with_ante_sent_added = False
            new_context = sentence["newContext"]
            ana_sent_id = int(sentence["anaphor"]["sentid"])
            ana_head = sentence["anaphor"]["head"]
            ana = sentence["anaphor"]["mentionStr"]
            new_context, num_ant_two_masks = self._mask_adj( new_context, num_ant_two_masks,_mask)
            options = sentence["options"]
            antecedents = sentence["antecedents"]
            ante_sent_ids = [int(ante["sentid"]) for ante in antecedents]
            # distances = [ana_sent_id-ante_sent_id for ante_sent_id in ante_sent_ids]

            nearest_ante_sent = None
            nearest_ante_dist = 10000000
            is_ante_same_sent = False
            is_ante_salient = False
            for ante_sent_id in ante_sent_ids:
                if ana_sent_id == ante_sent_id:
                    is_ante_same_sent = True
                    nearest_ante_sent = None
                    nearest_ante_dist = 0
                    break
                if ante_sent_id == 0:
                    is_ante_salient = True
                dist = ana_sent_id-ante_sent_id
                assert dist > -1
                if dist < nearest_ante_dist:
                    nearest_ante_dist = dist
                    is_sent_found = False
                    for s in sentence["sentences"]:
                        if s['id'] == ante_sent_id:
                            nearest_ante_sent = s["content"]
                            is_sent_found = True
                            break
                    assert is_sent_found

            sent_containing_anaphor = sentence["sentences"][-1]["content"]
            assert ana_head in sent_containing_anaphor,"[{}] no in [{}]".format(ana_head,sent_containing_anaphor)

            ana_mask = ana.replace(ana_head, "{} {}".format(ana_head,_mask))
            sent_anaphora_with_mask = sent_containing_anaphor.replace(ana,ana_mask)
            sent_anaphora_with_mask, _num_anas = self._mask_adj(sent_anaphora_with_mask, 0,_mask)
            # assert _num_anas == 0,"[{}] contains two anaphors [{}]".format(sent_anaphora_with_mask,ana)
            logger.debug("[{}] ---> [{}] ".format(sent_containing_anaphor,sent_anaphora_with_mask))

            sent_containing_anaphor_ante = None
            scores_with_ante_sent_added = 0
            is_correct_with_ante_sent_added = False
            sel_head_with_ante_sent_added = 0

            if not is_ante_same_sent:
                sent_containing_anaphor_ante = "{} {}".format(nearest_ante_sent,sent_anaphora_with_mask)
                total_ante_sent_added += 1
            ante_heads = [d[head_key] for d in antecedents]
            # print("heads : {} ".format(ante_heads))
            try:
                if not is_with_pert:
                    logger.debug("getting accuracy witout any context")
                    is_without_any_context_correct, scores_without_any_context_correct, sel_head_without_any_context,_, _, _, _, correct_without_any_context = self._get_target_acc(
                        counter, ana_mask, antecedents, options, nlp, ante_heads, num_cand_ant, num_score_null,
                        num_head_abs, num_error, correct_without_any_context)

                logger.debug("getting accuracy with context")
                is_with_context_correct,scores_with_context_correct,sel_head_with_context,num_cand_ant, num_score_null, num_head_abs, num_error, correct_with_context = self._get_target_acc(counter, new_context, antecedents, options, nlp, ante_heads,num_cand_ant, num_score_null, num_head_abs, num_error, correct_with_context)
                logger.debug("getting accuracy with only sentence with anaphor")
                is_correct_only_with_anaphor_sent, scores_only_with_anaphor_sent, sel_head_only_with_anaphor_sent, _, _, _, _, correct_only_with_anaphor_sent = self._get_target_acc(
                    counter, sent_anaphora_with_mask, antecedents, options, nlp, ante_heads, num_cand_ant, num_score_null,
                    num_head_abs, num_error, correct_only_with_anaphor_sent)
                if not is_ante_same_sent:
                    logger.debug("getting accuracy with antecedent sentence in addition to anaphor sentence")
                    is_correct_with_ante_sent_added, scores_with_ante_sent_added, sel_head_with_ante_sent_added, _, _, _, _, correct_with_ante_sent_added = self._get_target_acc(
                        counter, sent_containing_anaphor_ante, antecedents, options, nlp, ante_heads, num_cand_ant, num_score_null,
                        num_head_abs, num_error, correct_with_ante_sent_added)


            finally :
                total += 1
                sentence['pred_scores_with_context_correct'] = scores_with_context_correct
                sentence['pred_scores_only_with_anaphor_sent'] = scores_only_with_anaphor_sent
                sentence['pred_scores_with_ante_sent_added'] = scores_with_ante_sent_added

                sentence['is_with_context_correct'] = is_with_context_correct
                sentence['is_correct_only_with_anaphor_sent'] = is_correct_only_with_anaphor_sent
                sentence['is_correct_with_ante_sent_added'] = is_correct_with_ante_sent_added

                sentence['is_ante_salient'] = is_ante_salient
                sentence['nearest_ante_dist'] = nearest_ante_dist
                sentence['is_ante_same_sent'] = is_ante_same_sent

                sentence['sel_head_with_context'] = sel_head_with_context
                sentence['sel_head_only_with_anaphor_sent'] = sel_head_only_with_anaphor_sent
                sentence['sel_head_with_ante_sent_added'] = sel_head_with_ante_sent_added

                sentence['is_without_any_context_correct'] = is_without_any_context_correct
                sentence['pred_scores_without_any_context'] = scores_without_any_context_correct
                sentence['sel_head_without_any_context'] = sel_head_without_any_context

                sentence["newContext"] = new_context
                sentence['nearest_ante_sent'] = nearest_ante_sent
                sentence['sent_anaphor'] = sent_anaphora_with_mask
                pred_jsons.append(sentence)
            if counter % 10 == 0 and counter !=0:
                print("remaining {}".format(len(json_obj)-counter))
                # break

        print("WITHOUT ANY CONTEXT : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                         correct_without_any_context,
                                                                                         correct_without_any_context * 100.0 / total))
        print("WITH CONTEXT OF 2 SENTS AND SALIENTS : total : {} , correct : {} and accuracy : {}".format(total,correct_with_context,correct_with_context*100.0/total))
        print("WITH ONLY ANAPHOR SENTENCE : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                                          correct_only_with_anaphor_sent,
                                                                                                          correct_only_with_anaphor_sent * 100.0 / total))
        print("WITH ADDITION OF ANTE SENTENCE : total : {} , correct : {} and accuracy : {}".format(total_ante_sent_added,
                                                                                                          correct_with_ante_sent_added,
                                                                                                          correct_with_ante_sent_added * 100.0 / total_ante_sent_added))
        print("total erros : {}, num two masks :{}, num of null scores : {} and number of heads absent : {}".format(num_error,
                                                                                                  num_ant_two_masks,
                                                                                                  num_score_null,
                                                                                                  num_head_abs))
        print("total candidate antecedents : {} and average {}".format(num_cand_ant,num_cand_ant/total))

        logger.critical("WITHOUT ANY CONTEXT : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                                   correct_without_any_context,
                                                                                                   correct_without_any_context * 100.0 / total))
        logger.critical(
            "WITH CONTEXT OF 2 SENTS AND SALIENTS : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                                        correct_with_context,
                                                                                                        correct_with_context * 100.0 / total))
        logger.critical("WITH ONLY ANAPHOR SENTENCE : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                                          correct_only_with_anaphor_sent,
                                                                                                          correct_only_with_anaphor_sent * 100.0 / total))
        logger.critical(
            "WITH ADDITION OF ANTE SENTENCE : total : {} , correct : {} and accuracy : {}".format(total_ante_sent_added,
                                                                                                  correct_with_ante_sent_added,
                                                                                                  correct_with_ante_sent_added * 100.0 / total_ante_sent_added))
        logger.critical(
            "total erros : {}, num two masks :{}, num of null scores : {} and number of heads absent : {}".format(
                num_error,
                num_ant_two_masks,
                num_score_null,
                num_head_abs))
        logger.critical("total candidate antecedents : {} and average {}".format(num_cand_ant, num_cand_ant / total))

        _ext = "_with_pert" if is_with_pert else ""
        _file_ext = "_wo_of"+_ext if is_without_of else ""+_ext
        target_json_path += "_"+lm_model+_file_ext
        write_jsonlines_file(pred_jsons,target_json_path)
        # self.analyze_preds_with_entropy(pred_jsons)
        return pred_jsons


    def get_target_prob_acc_without_any_context(self,json_obj,target_json_path):
        from transformers import pipeline

        total = 0
        correct_without_any_context = 0

        nlp = pipeline("fill-mask", model=lm_model)
        num_cand_ant = 0
        num_ant_two_masks = 0
        num_error = 0
        num_score_null, num_head_abs = 0, 0
        pred_jsons = []
        for counter,sentence in enumerate(json_obj):
            logger.debug(5 * "*******")
            logger.debug("counter : {}".format(counter))
            logger.debug("*******")
            new_context = sentence["newContext"]
            ana_head = sentence["anaphor"]["head"]
            ana = sentence["anaphor"]["mentionStr"]
            new_context, num_ant_two_masks = self._mask_adj( new_context, num_ant_two_masks)
            options = sentence["options"]
            antecedents = sentence["antecedents"]
            ante_heads = [d[head_key] for d in antecedents]

            ana_mask = ana.replace(ana_head, "{} of [MASK]".format(ana_head))
            try:
                is_without_any_context_correct, scores_without_any_context_correct, sel_head_without_any_context, num_cand_ant, num_score_null, num_head_abs, num_error, correct_without_any_context = self._get_target_acc(
                    counter, ana_mask, antecedents, options, nlp, ante_heads, num_cand_ant, num_score_null,
                    num_head_abs, num_error, correct_without_any_context)


            finally :
                total += 1
                sentence['is_without_any_context_correct'] = is_without_any_context_correct
                sentence['pred_scores_without_any_context'] = scores_without_any_context_correct
                sentence['sel_head_without_any_context'] = sel_head_without_any_context
                pred_jsons.append(sentence)

            if counter % 10 == 0 and counter !=0:
                print("remaining {}".format(len(json_obj)-counter))
                # break


        print("WITHOUT ANY CONTEXT : total : {} , correct : {} and accuracy : {}".format(total,correct_without_any_context,correct_without_any_context*100.0/total))
        print("total erros : {}, num two masks :{}, num of null scores : {} and number of heads absent : {}".format(num_error,
                                                                                                  num_ant_two_masks,
                                                                                                  num_score_null,
                                                                                                  num_head_abs))
        print("total candidate antecedents : {} and average {}".format(num_cand_ant,num_cand_ant/total))

        logger.critical("WITHOUT ANY CONTEXT : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                                   correct_without_any_context,
                                                                                                   correct_without_any_context * 100.0 / total))
        logger.critical(
            "total erros : {}, num two masks :{}, num of null scores : {} and number of heads absent : {}".format(
                num_error,
                num_ant_two_masks,
                num_score_null,
                num_head_abs))
        logger.critical("total candidate antecedents : {} and average {}".format(num_cand_ant, num_cand_ant / total))

        write_jsonlines_file(pred_jsons,target_json_path)
        return pred_jsons



    def get_anaphor_head_dict(self):
        json_path = os.path.join(is_notes_data_path, is_notes_jsonlines)
        json_objects = read_jsonlines_file(json_path)
        ana_to_head_dict = {}
        repeated_ana = 0
        for json_object in json_objects:
            mention_ops = BridgingJsonDocInformationExtractor(json_object, logger)
            if mention_ops.is_file_empty:
                continue
            anaphor_words = mention_ops.get_span_words(mention_ops.anaphors)
            anaphors_heads, _ = mention_ops.get_spans_head(mention_ops.anaphors, True)
            for anaphor_word, anaphors_head in zip(anaphor_words, anaphors_heads):
                if ana_to_head_dict.get(anaphor_word, None) is not None:
                    # logger.debug("--")
                    # logger.debug("anaphor : [{}] exists with head [{}] and new head is {} ".format(anaphor_word,
                    #                                                                                   ana_to_head_dict.get(
                    #                                                                                       anaphor_word,
                    #                                                                                       None),
                    #                                                                                   anaphors_head))
                    try:
                        assert ana_to_head_dict.get(anaphor_word, None) == anaphors_head
                    except:
                        logger.critical("--")
                        logger.critical("new head , old head not same")
                        logger.critical("anaphor : [{}] exists with head [{}] and new head is {} ".format(anaphor_word,
                                                                                                       ana_to_head_dict.get(
                                                                                                           anaphor_word,
                                                                                                           None),
                                                                                                       anaphors_head))
                    repeated_ana += 1
                else:
                    ana_to_head_dict[anaphor_word] = anaphors_head

        assert len(ana_to_head_dict) == 663 - repeated_ana
        ana_to_head_dict['writer ? producers'] = 'producers'
        ana_to_head_dict['the state senate'] = 'senate'
        ana_to_head_dict['the structure'] = 'structure'
        ana_to_head_dict['The text by Patrick OConnor'] = 'text'
        ana_to_head_dict['reader'] = 'readers'
        ana_to_head_dict['even those countries that could nt produce more'] = 'countries'
        ana_to_head_dict['the winner'] = 'winner'
        ana_to_head_dict['North or South Korean residents'] = 'residents'
        ana_to_head_dict['the North Korean residents'] = 'residents'
        ana_to_head_dict['the South Korean residents'] = 'residents'
        ana_to_head_dict['the city'] = 'city'
        ana_to_head_dict["resident Joan OShea , who works in an acupuncturist 's office"] = 'resident'
        ana_to_head_dict['a resident who was nt allowed to go back inside'] = 'resident'
        ana_to_head_dict['the executive director'] = 'director'
        ana_to_head_dict['the shouting of `` Mort ! Mort ! Mort !'] = 'shouting'
        return ana_to_head_dict

    def get_anaphor_head(self,anaphor, cand_ante_list):
        if self.ana_to_head_dict.get(anaphor,None) is None:
            logger.critical("no head for {}".format(anaphor))
            self.num_ana_missed += 1
            ana_list = anaphor.split()
            logger.debug("anaphor : {}".format(anaphor))
            # valid_ana_words = [a for a in ana_list if a not in cand_ante_list]
            valid_ana_words = ana_list
            return random.choice(valid_ana_words)
        else:
            return self.ana_to_head_dict[anaphor]

    def create_sentences_from_json(self,json_obj):
        sentences = []
        labels_per_sentence = []
        for single_json_object in json_obj:
            logger.debug(5*"----")
            cont = single_json_object["OrigContext"]
            logger.debug("context : {}".format(cont))

            anaphor = single_json_object["anaphor"]['mentionStr']
            ana_head = single_json_object["anaphor"]["head"]
            ante_list = [i['improvedHead'] for i in single_json_object["antecedents"]]
            cans = [s['improvedHead'] for s in single_json_object["options"]]
            cans = list(set(cans))


            logger.debug("antecedent heads : {}".format(ante_list))
            logger.debug("candidate antecedent heads : {}".format(cans))

            # ana_head = self.get_anaphor_head(anaphor, cans)
            logger.debug("anaphor : [{}] head is [{}]".format(anaphor,ana_head))

            cont_list = cont.split()
            ana_head_ind = max([i for i, x in enumerate(cont_list) if x == ana_head])

            cand_ante_occurances = []
            for c in cans:
                _occ_ind = [i for i, x in enumerate(cont_list) if x == c]
                if c in ana_head:
                    _occ_ind = _occ_ind[0:-1]
                cand_ante_occurances.append(_occ_ind)

            ind_to_head = {}
            for i, occ_indx in enumerate(cand_ante_occurances):
                logger.debug("{}-{}".format(cans[i], occ_indx))
                for x in occ_indx:
                    if x < ana_head_ind:
                        assert ind_to_head.get(x, None) is None
                        ind_to_head[x] = cans[i]

            ordered_ind_to_head = collections.OrderedDict(sorted(ind_to_head.items()))
            labels = []
            inds = []
            for k, v in ordered_ind_to_head.items():
                inds.append(k)
                if v in ante_list:
                    labels.append(1)
                else:
                    labels.append(0)
                logger.debug(2* "==")
                logger.debug("{}-{}:{}".format(k, v, labels[-1]))

            assert ana_head_ind > max(ordered_ind_to_head.keys())

            logger.debug(cont)
            # cans.append(ana_head)
            # # cans.sort()
            # for c in cans:
            #     cont = cont.replace(" {} ".format(c), ' {} {} {} '.format(span_start_indi,c,span_end_indi))
            # logger.debug(3*"===")
            # logger.debug("\n")

            inds.append(ana_head_ind)
            inds.sort()
            for i, ind in enumerate(inds):
                if ind > ana_head_ind:
                    break
                ind = ind + 2 * i
                # print("---")
                # print(cont_list[ind])
                cont_list.insert(ind, span_start_indi)
                cont_list.insert(ind + 2, span_end_indi)
                # print(cont_list)

            # print(cont)
            # print(" ".join(cont_list))
            cont = " ".join(cont_list)
            logger.debug(cont)

            sentences.append(cont)
            labels_per_sentence.append(labels)

            cont_list = cont.split()
            num_start_span = len([i for i, x in enumerate(cont_list) if x == span_start_indi])
            num_end_span = len([i for i, x in enumerate(cont_list) if x == span_end_indi])

            try:
                assert num_start_span == num_end_span
                assert num_start_span == num_end_span == len(labels) +1
            except:
                logger.critical("\n \n")
                logger.critical("number of start, end span is not equal to labels ")
                logger.critical("number of start {}, end {} and labels : {}".format(num_start_span,num_end_span,len(labels)))
                raise AssertionError
        assert len(sentences) == len(labels_per_sentence) == 531
        return sentences,labels_per_sentence

    def get_accuracy_from_prom_heads(self,sentences,complete_sents_attn_weights,text_tokens,input_ids, labels_per_sentence):
        # marker_indices = self.find_all_mention_start_end_indicators(text_tokens)
        # print(complete_sents_attn_weights[0].shape)

        print("\n")
        for i in range(5):
            print("--- {}".format(i))
            print(sentences[i])
            print(text_tokens[i])

        # assert len(sentences) == len(complete_sents_attn_weights) == len(text_tokens),"{} {} {}".format(len(sentences),len(complete_sents_attn_weights),len(text_tokens))
        prom_heads= [[4,0],[8,11],[10,2]]
        num_pairs = 0
        correct_pairs = 0

        logger.debug(5*"******")
        logger.debug("getting accuracy")
        logger.debug(5 * "******")
        for i, sent in enumerate(sentences):
            num_pairs += 1
            logger.debug(3*"==**==")
            logger.debug("anaphor counter : {}".format(i))
            marker_index = self.find_all_mention_start_end_indicators([text_tokens[i]])[0]

            candidates_location = []
            for j in range(len(marker_index)-1):
                ant_s = marker_index[j][0][-1] + 1
                ant_e = marker_index[j][1][0]
                candidates_location.append((ant_s,ant_e))

            try:
                assert len(candidates_location) == len(labels_per_sentence[i])
            except:
                logger.critical("===")
                logger.critical("candidates : {} are not equal to labels : {}".format(len(candidates_location),len(labels_per_sentence[i])))
                logger.critical("cont : [{}]".format(text_tokens[i]))
                logger.critical("orig cont : [{}]".format(sentences[i]))
                logger.critical("candidate locations : {}".format(candidates_location))
                logger.critical("marked indices : {}".format(marker_index))
                logger.critical("number of marked indices : {}".format(len(marker_index)))
                logger.critical("labels : {}".format(labels_per_sentence[i]))

                raise AssertionError


            ana_s = marker_index[-1][0][-1] + 1
            ana_e = marker_index[-1][1][0]
            wts_per_cand = []

            for (ant_s,ant_e) in candidates_location:
                weight = 0
                for ana_ind in range(ana_s, ana_e + 1):
                    for l_h in prom_heads:
                        layer, attn = l_h
                        weight += sum(complete_sents_attn_weights[layer][i][attn][ana_ind][ant_s:ant_e + 1].tolist())
                wts_per_cand.append(weight)
            wts_per_cand = np.array(wts_per_cand)
            logger.debug("wts {}".format(wts_per_cand))
            max_ind = np.argmax(wts_per_cand)
            logger.debug("max at {}".format(max_ind))
            logger.debug("labels length {}".format(len(labels_per_sentence[i])))
            if labels_per_sentence[i][max_ind] == 1:
                correct_pairs +=1

        return correct_pairs


    def get_prom_head_based_accuracy(self):
        """If the prominent heads assign highest weights to correct antecedent amongst all candidate
        antecedents we say that its accuractely identified.
        input is json object containing context, anaphor, antecedent and candidate antecedents for each
        anaphor.
        """
        self.num_ana_missed = 0
        prob_data_2sents_path = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset_AllCandi_2Sent_new_27102020.json"
        json_data_2sents_obj = self.read_prob_dataset(prob_data_2sents_path)
        total = len(json_data_2sents_obj)
        assert total == 531

        # self.ana_to_head_dict = self.get_anaphor_head_dict()

        sentences, labels_per_sentence = self.create_sentences_from_json(json_data_2sents_obj)
        complete_sents_attn_weights, text_tokens,input_ids = self.get_bert_attn_weights(sentences)

        for i in range(5):
            print("--- {}".format(i))
            print(sentences[i])
            print(text_tokens[i])

        correct = self.get_accuracy_from_prom_heads(sentences,complete_sents_attn_weights,text_tokens,input_ids, labels_per_sentence)
        print("ana missed {}".format(self.num_ana_missed))
        logger.debug(5*"===")
        logger.debug("final result")
        logger.debug(5 * "===")
        logger.debug("total : {} , correct : {} and accuracy : {}".format(total,correct,correct*100.0/total))
        logger.debug("accuracy from total anaphors {}".format(correct*100.0/663))


    def analyze_all_sent_context_imp(self,target_file_path):
        def check_if_boolean(flag):
            assert (flag is True) or (flag is False)
        json_obj = read_jsonlines_file(target_file_path)

        dist_to_correct_number_dict = {}
        dist_to_total_number_dict = {}

        total = 0
        total_correct_ana = 0
        total_ante_sent_added = 0

        correct_with_context = 0
        correct_without_any_context = 0
        correct_only_with_anaphor_sent = 0
        correct_with_ante_sent_added = 0

        # anatecedent present in the anaphor sentence
        ap_T = 0
        ap_F = 0

        ca_T_T = 0
        ca_T_F = 0

        ca_F_F = 0
        ca_F_T = 0

        ap_T_T = []
        ap_T_F = []

        ap_F_F = []
        ap_F_T = []

        # antecedent absent in the anaphor sentence

        aa_F = 0
        aa_T = 0

        aa_F_ante_cont_add_F_F = 0
        aa_F_ante_cont_add_F_T = 0
        aa_F_ante_cont_add_T_F = 0
        aa_F_ante_cont_add_T_T = 0

        aa_T_ante_cont_add_F_F = 0
        aa_T_ante_cont_add_F_T = 0
        aa_T_ante_cont_add_T_F = 0
        aa_T_ante_cont_add_T_T = 0

        aa_F_F_F = []
        aa_F_F_T = []
        aa_F_T_F = []
        aa_F_T_T = []

        aa_T_F_F = []
        aa_T_F_T = []
        aa_T_T_F = []
        aa_T_T_T = []

        new_json = []
        for counter, sentence in enumerate(json_obj):
            total += 1
            ana_present_is_correct = None # if antedent is present in the same sentence as anaphor and result
            ana_absent_is_correct = None
            ana_ante_added_is_correct = None

            is_ante_same_sent = sentence['is_ante_same_sent']

            is_without_any_context_correct = sentence['is_without_any_context_correct']
            is_correct_only_with_anaphor_sent = sentence['is_correct_only_with_anaphor_sent']
            is_correct_with_ante_sent_added = sentence['is_correct_with_ante_sent_added']
            is_with_context_correct = sentence['is_with_context_correct']

            check_if_boolean(is_without_any_context_correct)
            if is_without_any_context_correct:
                correct_without_any_context+=1

            check_if_boolean(is_correct_only_with_anaphor_sent)
            if is_correct_only_with_anaphor_sent:
                correct_only_with_anaphor_sent+=1

            try:
                check_if_boolean(is_correct_with_ante_sent_added)
                if is_correct_with_ante_sent_added:
                    correct_with_ante_sent_added+=1
                total_ante_sent_added += 1
            except:
                logger.debug("is_correct_with_ante_sent_added : {}, not boolean".format(is_correct_with_ante_sent_added))

            check_if_boolean(is_with_context_correct)
            if is_with_context_correct:
                correct_with_context+=1

            is_ante_salient = sentence['is_ante_salient']
            nearest_ante_dist = sentence['nearest_ante_dist']

            if is_ante_same_sent:
                # total_ante_sent_added += 1
                ana_present_is_correct = is_correct_only_with_anaphor_sent
                if is_correct_only_with_anaphor_sent:
                    # correct_with_ante_sent_added += 1
                    ap_T += 1
                    if is_with_context_correct:
                        ca_T_T += 1
                        ap_T_T.append(counter)
                    else:
                        ca_T_F += 1
                        ap_T_F.append(counter)
                else:
                    ap_F += 1
                    if is_with_context_correct:
                        ca_F_T += 1
                        ap_F_T.append(counter)
                    else:
                        ca_F_F += 1
                        ap_F_F.append(counter)
            else:
                ana_absent_is_correct = is_correct_only_with_anaphor_sent
                ana_ante_added_is_correct = is_correct_with_ante_sent_added

                if is_correct_only_with_anaphor_sent:
                    aa_T += 1
                    if is_correct_with_ante_sent_added:
                        if is_with_context_correct:
                            aa_T_ante_cont_add_T_T += 1
                            aa_T_T_T.append(counter)
                        else:
                            aa_T_ante_cont_add_T_F += 1
                            aa_T_T_F.append(counter)
                    else:
                        if is_with_context_correct:
                            aa_T_ante_cont_add_F_T += 1
                            aa_T_F_T.append(counter)
                        else:
                            aa_T_ante_cont_add_F_F += 1
                            aa_T_F_F.append(counter)

                else:
                    aa_F += 1
                    if is_correct_with_ante_sent_added:
                        if is_with_context_correct:
                            aa_F_ante_cont_add_T_T += 1
                            aa_F_T_T.append(counter)
                            if int(sentence['nearest_ante_dist']) > 2:
                                logger.critical(3 * "***")
                                logger.critical("pair is corrected after addition of context but antecedent not present")
                                self._log_sentence_context_info([counter],json_obj)
                                logger.critical(3*"***")
                        else:
                            aa_F_ante_cont_add_T_F += 1
                            aa_F_T_F.append(counter)
                            if int(sentence['nearest_ante_dist']) < 2:
                                logger.critical(3 * "***")
                                logger.critical("pair is not corrected after addition of context but antecedent is present in the context")
                                self._log_sentence_context_info([counter],json_obj)
                                logger.critical(3*"***")
                    else:
                        if is_with_context_correct:
                            aa_F_ante_cont_add_F_T += 1
                            aa_F_F_T.append(counter)
                            if int(sentence['nearest_ante_dist']) > 2:
                                logger.critical(3 * "***")
                                logger.critical("pair is corrected after addition of context but antecedent not present")
                                self._log_sentence_context_info([counter],json_obj)
                                logger.critical(3*"***")
                        else:
                            aa_F_ante_cont_add_F_F += 1
                            aa_F_F_F.append(counter)



            context_2sent_salient_is_correct = is_with_context_correct

            dist = -1 if is_ante_salient else nearest_ante_dist
            _add = 1 if is_with_context_correct else 0

            total_correct_ana += _add
            dist_to_total_number_dict[dist] = dist_to_total_number_dict.get(dist, 0) + 1
            dist_to_correct_number_dict[dist] = dist_to_correct_number_dict.get(dist, 0) + _add

            sentence['ana_present_is_correct'] = ana_present_is_correct
            sentence['ana_absent_is_correct'] = ana_absent_is_correct
            sentence['ana_ante_added_is_correct'] = ana_ante_added_is_correct
            sentence['context_2sent_salient_is_correct'] = context_2sent_salient_is_correct

            new_json.append(sentence)



        print("WITHOUT ANY CONTEXT : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                         correct_without_any_context,
                                                                                         correct_without_any_context * 100.0 / total))
        print("WITH CONTEXT OF 2 SENTS AND SALIENTS : total : {} , correct : {} and accuracy : {}".format(total,correct_with_context,correct_with_context*100.0/total))
        print("WITH ONLY ANAPHOR SENTENCE : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                                          correct_only_with_anaphor_sent,
                                                                                                          correct_only_with_anaphor_sent * 100.0 / total))
        print("WITH ADDITION OF ANTE SENTENCE : total : {} , correct : {} and accuracy : {}".format(total_ante_sent_added,
                                                                                                          correct_with_ante_sent_added,
                                                                                                          correct_with_ante_sent_added * 100.0 / total_ante_sent_added))


        logger.critical("WITHOUT ANY CONTEXT : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                                   correct_without_any_context,
                                                                                                   correct_without_any_context * 100.0 / total))
        logger.critical(
            "WITH CONTEXT OF 2 SENTS AND SALIENTS : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                                        correct_with_context,
                                                                                                        correct_with_context * 100.0 / total))
        logger.critical("WITH ONLY ANAPHOR SENTENCE : total : {} , correct : {} and accuracy : {}".format(total,
                                                                                                          correct_only_with_anaphor_sent,
                                                                                                          correct_only_with_anaphor_sent * 100.0 / total))
        logger.critical(
            "WITH ADDITION OF ANTE SENTENCE : total : {} , correct : {} and accuracy : {}".format(total_ante_sent_added,
                                                                                                  correct_with_ante_sent_added,
                                                                                                  correct_with_ante_sent_added * 100.0 / total_ante_sent_added))


        logger.critical(5*"*******")
        logger.critical("Antecedents present in the anaphor sentence ")
        logger.critical(5 * "---")

        assert ap_T == ca_T_F + ca_T_T
        logger.critical("{}(T)     {}(TT)      {}(TF)".format(ap_T,ca_T_T,ca_T_F))

        assert ap_F == ca_F_T + ca_F_F
        logger.critical("{}(F)     {}(FT)       {}(FF)".format(ap_F,ca_F_T,ca_F_F))
        logger.critical(5 * "+++++")

        logger.critical("Antecedent absent in the anaphor sentence ")
        logger.critical(5 * "---")

        assert aa_F == aa_F_ante_cont_add_F_F + aa_F_ante_cont_add_F_T + aa_F_ante_cont_add_T_F + aa_F_ante_cont_add_T_T
        logger.critical("{}(F)     {}(FF)       {}(FT)     {}(TF)       {}(TT)".format(aa_F,aa_F_ante_cont_add_F_F ,aa_F_ante_cont_add_F_T ,aa_F_ante_cont_add_T_F ,aa_F_ante_cont_add_T_T))

        logger.critical(5 * "+++++")
        assert aa_T == aa_T_ante_cont_add_F_F + aa_T_ante_cont_add_F_T + aa_T_ante_cont_add_T_F + aa_T_ante_cont_add_T_T
        logger.critical("{}(T)     {}(FF)       {}(FT)     {}(TF)       {}(TT)".format(aa_T, aa_T_ante_cont_add_F_F,
                                                                                       aa_T_ante_cont_add_F_T,
                                                                                       aa_T_ante_cont_add_T_F,
                                                                                       aa_T_ante_cont_add_T_T))

        assert aa_T + aa_F + ap_T + ap_F == 622,"{} is not 622".format(aa_T + aa_F + ap_T + ap_F)

        assert len(dist_to_total_number_dict) == len(dist_to_correct_number_dict),"total : {} \n correct : {}".format(dist_to_total_number_dict,dist_to_correct_number_dict)

        logger.critical(5 * "*******")
        logger.critical("total anaphors {} correctly linked {} and percentage accuracy {}".format(total,total_correct_ana,total_correct_ana/total))
        logger.critical(5 * "+++++")
        logger.critical("Accuracies by distance between anaphor and antecedent")

        dist_to_perc_acc_dict = {}
        t,c = 0,0
        for k in sorted(dist_to_total_number_dict):
            if k < 3:
                dist_to_perc_acc_dict[k] = float(dist_to_correct_number_dict[k])/dist_to_total_number_dict[k]
                logger.critical(2*"---")
                logger.critical("distance : {} , total : {} , correct : {} and perce {}".format(k,dist_to_total_number_dict[k],dist_to_correct_number_dict[k],dist_to_perc_acc_dict[k]))
            else:
                t += dist_to_total_number_dict[k]
                c += dist_to_correct_number_dict[k]

        logger.critical(2 * "---")
        logger.critical("distance : >2 , total : {} , correct : {} and perce {}".format(t,c,c/t))

        logger.critical(5 * "***************")
        write_jsonlines_file(new_json, target_file_path+"_analyzed")

        logger.critical(5 * "***************")
        logger.critical("Antecedent is present in the anaphor sentence")
        logger.critical(5 * "***************")
        logger.critical("CASE 1 : RESULT REMAINS SAME")
        logger.critical(5 * "~~~~~~~")
        logger.critical("Correct remains correct")
        self._log_sentence_context_info(ap_T_T,json_obj)
        logger.critical(5* "~~~~~~~")
        logger.critical("Incorrect remains incorrect")
        self._log_sentence_context_info(ap_F_F, json_obj)
        logger.critical(5 * "***************")
        logger.critical("CASE 2 : RESULT CHANGES")
        logger.critical(5 * "~~~~~~~")
        logger.critical("Incorrect is corrected")
        self._log_sentence_context_info(ap_F_T,json_obj)
        logger.critical(5* "~~~~~~~")
        logger.critical("Correct becomes incorrect")
        self._log_sentence_context_info(ap_T_F, json_obj)
        logger.critical(5 * "***************")

        logger.critical(5 * "***************")
        logger.critical("Antecedent is absent from the anaphor sentence")
        logger.critical(5 * "***************")

        logger.critical("CASE 1 : RESULT REMAINS SAME FOR BOTH TYPE OF CONTEXT ADDITION")
        logger.critical(5 * "~~~~~~~")
        logger.critical("Correct remains correct")
        self._log_sentence_context_info(aa_T_T_T,json_obj)
        logger.critical(5* "~~~~~~~")
        logger.critical("Incorrect remains incorrect")
        self._log_sentence_context_info(aa_F_F_F, json_obj)
        logger.critical(5 * "***************")

        logger.critical("CASE 2 : RESULT CHANGES FOR BOTH TYPE OF CONTEXT ADDITION")
        logger.critical(5 * "~~~~~~~")
        logger.critical("Incorrect is corrected")
        self._log_sentence_context_info(aa_F_T_T,json_obj)
        logger.critical(5* "~~~~~~~")
        logger.critical("Correct becomes incorrect")
        self._log_sentence_context_info(aa_T_F_F, json_obj)
        logger.critical(5 * "***************")


        logger.critical("CASE 3: RESULT IMPROVES WITH EITHER WITH ONE")
        logger.critical(5 * "~~~~~~~")
        logger.critical("Incorrect is corrected with addition of antecedent sentence but not with context")
        self._log_sentence_context_info(aa_F_T_F,json_obj)
        logger.critical(5* "~~~~~~~")
        logger.critical("Incorrect is corrected with addition of context but not with antecedent sentence")
        self._log_sentence_context_info(aa_F_F_T, json_obj)
        logger.critical(5 * "***************")

        logger.critical("CASE 3: RESULT DEGRADES WITH EITHER WITH ONE")
        logger.critical(5 * "~~~~~~~")
        logger.critical("Correct is incorrected with addition of antecedent sentence but not with context")
        self._log_sentence_context_info(aa_T_F_T,json_obj)
        logger.critical(5* "~~~~~~~")
        logger.critical("Correct is incorrected with addition of context but not with antecedent sentence")
        self._log_sentence_context_info(aa_T_T_F, json_obj)
        logger.critical(5 * "***************")



    def _log_sentence_context_info(self,indices,json_obj):
        for c,i in enumerate(indices):
            sentence = json_obj[i]
            logger.critical("----")
            logger.critical("sent : {} \n".format(json_obj[i]['newContext']))
            logger.critical("ana : {} \n".format(json_obj[i]['anaphor']))
            logger.critical("antecedents : {} \n".format(json_obj[i]['antecedents']))
            logger.critical("head selection : only with anaphor : [{}], with addition of antecedent sent : [{}], with addition of context : [{}] \n".format(sentence['sel_head_only_with_anaphor_sent'],sentence['sel_head_with_ante_sent_added'],sentence['sel_head_with_context']))
            logger.critical("RESULT : only with anaphor : {}, after adding antecedent sentence : {}, after adding context : {}. \n".format(sentence['is_correct_only_with_anaphor_sent'],sentence['is_correct_with_ante_sent_added'],sentence['is_with_context_correct']))
            logger.critical("anaphor sentence : [{}] \n antecedent sentence : [{}] \n ".format(sentence['sent_anaphor'],sentence['nearest_ante_sent']))











if __name__ == '__main__':
    ab = AnalyzeBERT(json_objects_path=is_notes_data_path,jsonlines=is_notes_jsonlines,is_head_sent=True)
    # ab.analyze_ana_ante_attn_weights_all_dist_sents()
    # ab.analyze_ana_ante_attn_weights()
    ab.analyze_ana_ante_attn_weights_all_dist_sents_prominent_heads()
    # prob_path = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset"
    # prob_path = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset_AllCandi"
    # prob_data_all_sents_path = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset_AllCandi_AllSent_new.json"
    # prob_data_all_sents_path = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset_AllCandi_AllSent_new_27102020.json"
    # prob_data_2sents_path = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset_AllCandi_2Sent_new.json"
    # prob_data_2sents_path = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset_AllCandi_2Sent_new_27102020.json"

    # prob_data_all_sents_path = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset_AllCandi_AllSent_new_04112020.json"
    # prob_data_all_sents_pred_path ="/srv/storage/magnet@storage1.lille.grid5000.fr/opandit/Data/bridging/probingDataset_AllCandi_AllSent_new_04112020.json_pred_bert-base-cased"
    # json_data_all_sents_obj = read_jsonlines_file(prob_data_all_sents_pred_path)
    # ab.get_target_prob_acc_without_any_context(json_data_all_sents_obj,prob_data_all_sents_pred_path)

    # json_data_2sents_obj = ab.read_prob_dataset(prob_data_2sents_path)
    # json_data_all_sents_obj = ab.read_prob_dataset(prob_data_all_sents_path)[60:80]
    # pred_file_ext = "_9_pred"
    # json_obj,un_common_jason_obj = ab.get_common_anaphors(json_data_all_sents_obj,json_data_2sents_obj)
    # ab.get_target_prob_accuracy_bert_compliant_sentences(un_common_jason_obj)
    # ab.get_target_prob_accuracy_bert_compliant_sentences(json_obj)
    # ab.get_target_prob_accuracy_bert_compliant_sentences(json_data_2sents_obj,prob_data_2sents_path+"_pred")
    # ab.get_target_prob_accuracy_bert_compliant_sentences(json_data_all_sents_obj,prob_data_all_sents_path+"_pred")
    # ab.get_target_prob_acc_depending_context(json_data_all_sents_obj, prob_data_all_sents_path + pred_file_ext,is_without_of=True,is_with_pert=True)
    # ab.get_prom_head_based_accuracy()
    # ab.analyze_preds_with_entropy(pred_jsons_path=prob_data_all_sents_path+"_pred_"+lm_model)
    # json_objects = []
    # for i in range(10):
    #     pred_file_ext = "_{}_pred_{}_wo_of".format(i,lm_model)
    #     json_objects += read_jsonlines_file(prob_data_all_sents_path + pred_file_ext)
    # assert len(json_objects) == 622
    # ids = []
    # for jo in json_objects:
    #     ids.append(jo["anaphor"]["id"])
    # ids = list(set(ids))
    # assert len(ids) == 622
    # pred_file_ext = "_pred_{}_wo_of".format(lm_model)
    # write_jsonlines_file(json_objects, prob_data_all_sents_path + pred_file_ext)
    # ab.analyze_all_sent_context_imp(prob_data_all_sents_path + pred_file_ext)
