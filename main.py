# -*- coding: utf-8 -*-
import numpy as np
import argparse

from get_datasets import load_dataset, interpolate
from plot_util import plot_acc_loss, plot_outputs, plot_activations
from util import *
from model import train, retrain
import json

def get_parser():
    '''
    Set up argument parser
    Returns:
        parser: (ArgumentParser) the created parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        action='store_true')
    parser.add_argument('--eval_only',
                        action='store_true')
    parser.add_argument('--listener',
                        required=True,
                        choices={'english', 'EN',
                                 'japanese', 'JN',
                                 'j_learner', 'JL',
                                 'bilingual', 'BL'})
    parser.add_argument('--num_epochs',
                        required=True,
                        type=int)
    parser.add_argument('--num_iterations',
                        required=True,
                        type=int)
    parser.add_argument('--num_layers',
                        required=True,
                        type=int)
    parser.add_argument('--num_dim',
                        required=True,
                        type=int)
    return parser

def save(model, iteration, stats, filename=''):
    model.save('%s.h5' % filename)

    ckpt, accs, losses = stats

    np.save('%s_accs.npy' % filename, np.array(accs))
    np.save('%s_losses.npy' % filename, np.array(losses))

    labels, data = ckpt.dump()

    np.save('%s_ckpt.npy' % filename, data)

    with open('%s.json' % filename, 'w') as f:
        obj = {}
        obj['true_labels'] = labels
        obj['best_iteration'] = iteration
        f.write(json.dumps(obj))

if __name__ == '__main__':
    # parse arguments
    parser = get_parser()
    args = parser.parse_args()
    listener = get_listener(args.listener)

    # get datasets
    J_data, E_data = load_dataset(listener)

    if listener == JAPANESE:
        # unpack data
        J_train_X, J_test_X, J_train_y, J_test_y = J_data
        E_test_X, E_test_y = E_data

        # train on japanese data
        model, i, stats = train(J_train_X, J_train_y,
                                test_X=J_test_X,
                                test_y=J_test_y,
                                num_epochs=args.num_epochs,
                                num_iterations=args.num_iterations,
                                num_layers=args.num_layers,
                                num_dim=args.num_dim)

        save(model, i, stats, filename='weights/jn_model_l%d_d%d' % (args.num_layers, args.num_dim))

        # unpack statistics
        ckpt, accs, losses = stats

        # plot training data
        plot_acc_loss(accs, losses, i,
                      listener=listener,
                      filename="data/plots/jn_acc_loss_l%d_d%d.png" % (args.num_layers, args.num_dim))

        # plot testing data
        labels = [u'ら', u'り', u'る', u'れ', u'ろ']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="data/plots/jn_jp_act_l%d_d%d.png" % (args.num_layers, args.num_dim))

    elif listener == ENGLISH:
        # unpack data
        E_train_X, E_test_X, E_train_y, E_test_y = E_data

        # train on english data
        model, i, stats = train(E_train_X, E_train_y,
                                test_X=E_test_X,
                                test_y=E_test_y,
                                num_epochs=args.num_epochs,
                                num_iterations=args.num_iterations,
                                num_layers=args.num_layers,
                                num_dim=args.num_dim)


        save(model, i, stats, filename='weights/en_model_l%d_d%d' % (args.num_layers, args.num_dim))

        # unpack statistics
        ckpt, accs, losses = stats

        # plot training data
        plot_acc_loss(accs, losses, i,
                      listener=listener,
                      filename="data/plots/en_acc_loss_l%d_d%d.png" % (args.num_layers, args.num_dim))

        # plot testing data
        labels = [u'la', u'li', u'lu', u'le', u'lo']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="data/plots/en_el_act_l%d_d%d.png" % (args.num_layers, args.num_dim))

        labels = [u'ra', u'ri', u'ru', u're', u'ro']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="data/plots/en_er_act_l%d_d%d.png" % (args.num_layers, args.num_dim))

    elif listener == JLEARNER:
        # unpack data
        J_train_X, J_test_X, J_train_y, J_test_y = J_data
        E_train_X, E_test_X, E_train_y, E_test_y = E_data

        # train on japanese data
        model, i, stats = train(J_train_X, J_train_y,
                                test_X=J_test_X,
                                test_y=J_test_y,
                                num_epochs=args.num_epochs,
                                num_iterations=args.num_iterations,
                                num_layers=args.num_layers,
                                num_dim=args.num_dim)

        save(model, i, stats, filename='weights/jl0_model_l%d_d%d' % (args.num_layers, args.num_dim))

        # unpack statistics
        ckpt, accs, losses = stats

        # plot training data
        plot_acc_loss(accs, losses, i,
                      listener=listener,
                      filename="data/plots/jl_jp_acc_loss_l%d_d%d.png" % (args.num_layers, args.num_dim))

        # plot testing data
        labels = [u'ら', u'り', u'る', u'れ', u'ろ']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="data/plots/jl_jp_act_l%d_d%d.png" % (args.num_layers, args.num_dim))

        # retrain on english data
        model, i, stats = retrain(model,
                                  E_train_X, E_train_y,
                                  test_X=E_test_X,
                                  test_y=E_test_y,
                                  num_epochs=args.num_epochs,
                                  num_iterations=args.num_iterations)

        save(model, i, stats, filename='weights/jl1_model_l%d_d%d' % (args.num_layers, args.num_dim))

        # unpack statistics
        ckpt, accs, losses = stats

        # plot training data
        plot_acc_loss(accs, losses, i,
                      listener=listener,
                      filename="data/plots/jl_en_acc_loss_l%d_d%d.png" % (args.num_layers, args.num_dim))

        # plot testing data
        labels = [u'la', u'li', u'lu', u'le', u'lo']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="data/plots/jl_el_act_l%d_d%d.png" % (args.num_layers, args.num_dim))

        labels = [u'ra', u'ri', u'ru', u're', u'ro']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="data/plots/jl_er_act_l%d_d%d.png" % (args.num_layers, args.num_dim))

    elif listener == BILINGUAL:
        # unpack data
        J_train_X, J_test_X, J_train_y, J_test_y = J_data
        E_train_X, E_test_X, E_train_y, E_test_y = E_data

        # interpolate data
        # JE_train_X, JE_train_y = interpolate(J_train_X, E_train_X, J_train_y, E_train_y)
        # JE_test_X, JE_test_y = interpolate(J_test_X, E_test_X, J_test_y, E_test_y)

        JE_train_X = np.vstack([J_train_X, E_train_X])
        JE_train_y = np.vstack([J_train_y, E_train_y])
        JE_test_X = np.vstack([J_test_X, E_test_X])
        JE_test_y = np.vstack([J_test_y, E_test_y])

        # train on japanese data
        model, i, stats = train(JE_train_X, JE_train_y,
                                test_X=JE_test_X,
                                test_y=JE_test_y,
                                num_epochs=args.num_epochs,
                                num_iterations=args.num_iterations,
                                num_layers=args.num_layers,
                                num_dim=args.num_dim)

        save(model, i, stats, filename='weights/bl_model_l%d_d%d' % (args.num_layers, args.num_dim))

        # unpack statistics
        ckpt, accs, losses = stats

        # plot training data
        plot_acc_loss(accs, losses, i,
                      listener=listener,
                      filename="data/plots/bl_je_acc_loss_l%d_d%d.png" % (args.num_layers, args.num_dim))

        # plot testing data
        labels = [u'ら', u'り', u'る', u'れ', u'ろ']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="data/plots/bl_jp_act_l%d_d%d.png" % (args.num_layers, args.num_dim))

        labels = [u'la', u'li', u'lu', u'le', u'lo']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="data/plots/bl_el_act_l%d_d%d.png" % (args.num_layers, args.num_dim))

        labels = [u'ra', u'ri', u'ru', u're', u'ro']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="data/plots/bl_er_act_l%d_d%d.png" % (args.num_layers, args.num_dim))
    else:
        # train on interpolated data
        # test on interpolated data
        # test on english data
        # test on japanese data
        raise NotImplementedError
