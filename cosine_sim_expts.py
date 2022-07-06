from common_code import *
import numpy as np
np.random.seed(1313)
from bridging_json_extract import BridgingJsonDocInformationExtractor
from bridging_utils import *
from evaluation_measures import *
from expt_params import ExptParams
from word_embeddings import EmbeddingsPP

class CosineSimilarityExpts(ExptParams):

    def evaluate_embeddings_pp(self):
        embeddings_pp = EmbeddingsPP(100, "", logger)
        json_objects = self.data.get_json_objects(self.jsonlines)
        cum_acc_pred, cum_total_pred, cum_total_gold = 0, 0, 0
        for single_json_object in json_objects:
            mention_ops = BridgingJsonDocInformationExtractor(single_json_object, logger)
            if mention_ops.is_file_empty:
                continue
            # ana_to_antecdents_map_selected_data = mention_ops.get_ana_to_anecedent_map_from_selected_data(False,False,False,False)
            cand_ante_per_anaphor = self.data.generate_cand_ante_by_sentence_window(mention_ops, sen_window_size=2, is_consider_saleint=False,
                                                                                    is_training=False,
                                                                                    ana_to_antecedent_map=None)
            predicted_pairs = []
            assert len(cand_ante_per_anaphor) == len(mention_ops.anaphors)
            for i, cand_antes in enumerate(cand_ante_per_anaphor):
                curr_ana = mention_ops.anaphors[i]
                logger.debug(3 * "----")
                logger.debug("anaphor {} candidate antecedents {} ".format(curr_ana, cand_antes))
                assert len(cand_antes) != 0
                anphor_words = mention_ops.get_words_for_start_end_indices(curr_ana)
                anphor_head_words = mention_ops.get_span_head(curr_ana)
                anphor_emb = embeddings_pp.get_vec_for_anphor(anphor_head_words, is_force=False).reshape(1, -1)
                # anphor_emb.reshape(1,-1)
                cosine_sims = []
                for cand_ante in cand_antes:
                    cand_ante_head_word = mention_ops.get_span_head(cand_ante)
                    cand_ante_emb = embeddings_pp.get_vec_for_word(cand_ante_head_word, is_force=False).reshape(1, -1)
                    # cand_ante_emb.reshape(1,-1)
                    sim = sklearn.metrics.pairwise.cosine_similarity(cand_ante_emb, anphor_emb)
                    # print(sim.shape)
                    # print(sim)
                    cosine_sims.append(sim[0][0])
                pred = np.array(cosine_sims)
                logger.debug("pred shape {}".format(pred.shape))
                logger.debug("predicted probs {}".format(pred))
                max_score_ind = np.argmax(pred)
                logger.debug("max score at {}".format(max_score_ind))
                pred_ante = cand_antes[max_score_ind]
                pair = (pred_ante, curr_ana)
                predicted_pairs.append(pair)
                logger.debug("pair added {}".format(pair))

            assert len(predicted_pairs) == len(mention_ops.bridging_clusters)

            accurate_pred, total_pred, total_gold, pred_pair_labels = get_accurate_pred_gold_numbers(mention_ops,
                                                                                                     predicted_pairs)
            error_analysis(mention_ops, predicted_pairs, pred_pair_labels)
            cum_acc_pred += accurate_pred
            cum_total_gold += total_gold
            cum_total_pred += total_pred
        acc = get_accuracy(cum_acc_pred, cum_total_pred, cum_total_gold)
        print(acc)
        print("total ana {} not found {}".format(embeddings_pp.total_anahors, embeddings_pp.total_anahors_not_found))
        print(
            "total anatecedents {} not found {}".format(embeddings_pp.total_words, embeddings_pp.total_not_found_words))
        return acc

    def cosine_simi_exp(self):
        data_params = self.get_data_param_comb(is_svm=True)
        null_hyper_params = ["", "", "", "", ""]

        results = []
        for param in data_params:
            param = param + null_hyper_params
            self.set_params(*param)
            expt_name = self.get_expt_name("cosine_sim")
            print(10 * "---")
            print("expt : {}".format(expt_name))
            print(10 * "---")

            logger.critical(10 * "---")
            logger.critical("expt : {}".format(expt_name))
            logger.critical(10 * "---")

            acc,_,_ = self.evaluate_svm(None, self.jsonlines)
            print("accuracy {}".format(acc))
            logger.critical("accuracy {}".format(acc))
            results.append("{} : {}".format(expt_name, acc))
        result_file = os.path.join(result_path, timestr + '_final.result.txt')
        write_text_file(result_file, results)




if __name__ == '__main__':
    bm = CosineSimilarityExpts()
    # bm.evaluate_embeddings_pp()
    # bm.cosine_simi_exp()
