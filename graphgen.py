import numpy as np
import time
from collections import defaultdict
import os
import torch
import torch.nn as nn
from torch.autograd import Variable

import math
from torch.optim import Optimizer

from clize import run

"""
from rdkit import Chem
s = 'Cc1cc(ccc1C(=O)c2ccccc2Cl)N3N=CC(=O)NC3=O'
a, b = s, deprocess(*preprocess(s))
print(a, b)
print(Chem.MolFromSmiles(a) == Chem.MolFromSmiles(b))
"""


cuda = True

def layernorm(h):
    mean = h.mean(1)
    std = h.std(1)
    mean = mean.repeat(1, h.size(1))
    std  = std.repeat(1, h.size(1))
    return (h - mean) / std

class RNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, n_edge_types, max_length=10, repr_size=100, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        vocab_size = input_size
        self.vertex_encoder = nn.Embedding(input_size, emb_size)
        self.vertex_rnn = nn.GRUCell(emb_size, hidden_size)
        self.vertex_out = nn.Linear(hidden_size, vocab_size)
        self.vertex_repr = nn.Linear(hidden_size, repr_size)

        self.edge_encoder = nn.Embedding(max_length, emb_size)
        self.edge_type_encoder = nn.Embedding(n_edge_types, emb_size)
        self.edge_rnn = nn.GRUCell(emb_size * 2, hidden_size)
        self.edge_repr = nn.Linear(hidden_size, repr_size)
        self.edge_type = nn.Linear(repr_size, n_edge_types)
        self.edge_out_src = nn.Linear(repr_size, 1)
        self.edge_out_dst = nn.Linear(repr_size, 1)
    
    def next_vertex(self, input_vertex, h):
        batch_size = input_vertex.size(0)
        V = self.vertex_encoder(input_vertex)
        V = V.transpose(0, 1)
        h = self.vertex_rnn(V[0], h)
        o = self.vertex_out(h)
        return o, h

    def next_edge(self, input_edge, h):
        batch_size = input_edge.size(0)
        E_src = self.edge_encoder(input_edge[:, :, 0])
        E_dst = self.edge_encoder(input_edge[:, :, 1])
        E = torch.cat((E_src, E_dst), 2)
        E = E.transpose(0, 1)
        h = self.edge_rnn(E[0], h)
        return h

    def forward(self, input_vertex, input_edge):
        batch_size = input_vertex.size(0)
        V = self.vertex_encoder(input_vertex)
        V = V.transpose(0, 1)
        e = []
        hv = []
        
        hi = Variable(torch.zeros(batch_size, self.hidden_size))
        if cuda: hi = hi.cuda()

        O = []
        for i in range(V.size(0)):
            hi = self.vertex_rnn(V[i], hi)
            hv.append(hi)
            oi = self.vertex_out(hi)
            oi = oi.view(oi.size(0), oi.size(1), 1)
            O.append(oi)
        O = torch.cat(O, 2)
        # nb_examples, vocab_size, nb_vertex_timesteps
        O_vertex = O

        E_src = self.edge_encoder(input_edge[:, :, 0])
        E_dst = self.edge_encoder(input_edge[:, :, 1])
        E_type = self.edge_type_encoder(input_edge[:, :, 2])
        E = torch.cat((E_src, E_dst), 2)
        E = E.transpose(0, 1)
        E_type = E_type.transpose(0, 1)
        d = []
        he = []
        
        hi = Variable(torch.zeros(batch_size, self.hidden_size))
        if cuda: hi = hi.cuda()
        O = []
        Ot = []
        for i in range(E.size(0)):
            hi = self.edge_rnn(E[i], hi)
            e = self.edge_repr(hi)
            t = self.edge_type(e)
            t =  t.view(t.size(0), t.size(1), 1)
            Ot.append(t)
            osrc = []
            odst = []
            for j in range(V.size(0)):
                v = self.vertex_repr(hv[j])
                u = nn.Tanh()(e + v)
                usrc = self.edge_out_src(u)
                udst = self.edge_out_dst(u)
                osrc.append(usrc)
                odst.append(udst)

            osrc = torch.cat(osrc, 1)
            osrc = osrc.view(osrc.size(0), osrc.size(1), 1)
            odst = torch.cat(odst, 1)
            odst = odst.view(odst.size(0), odst.size(1), 1)
            o = torch.cat((osrc, odst), 2)
            o = o.view(o.size(0), o.size(1), o.size(2), 1)
            O.append(o)

        O = torch.cat(O, 3)
        # examples, nb_vert_timesteps, 2(dst + src), nb_edge_timesteps
        O_edge = O
        Ot = torch.cat(Ot, 2)
        O_edge_type = Ot
        return O_vertex, O_edge, O_edge_type

def acc(pred, true):
    _, pred_classes = pred.max(1)
    acc = (pred_classes == true).float().mean()
    return acc

def generate(model, n_vertex_steps=10, n_edge_steps=10, out='out.png'):
    batch_size = 1
    h = Variable(torch.zeros(batch_size, model.hidden_size))
    v = Variable(torch.zeros(batch_size, 1).long())
    
    if cuda: h = h.cuda()
    if cuda: v = v.cuda()

    hv = []
    vertices = []
    for _ in range(n_vertex_steps):
        o, h = model.next_vertex(v, h)
        hv.append(h)
        o = nn.Softmax()(o)
        v = torch.multinomial(o)
        vertices.append(v.data[0, 0])
        if vertices[-1] == 0:
            break
    h = Variable(torch.zeros(batch_size, model.hidden_size))
    if cuda: h = h.cuda()
    E = Variable(torch.zeros(batch_size, 1, 3).long())
    if cuda: E = E.cuda()

    edges = []
    for _ in range(n_edge_steps):
        osrc = []
        odst = []
        h = model.next_edge(E, h)
        e = model.edge_repr(h)
        otype = model.edge_type(e)
        for j in range(len(vertices)):
            v = model.vertex_repr(hv[j])
            u = nn.Tanh()(e + v)
            usrc = model.edge_out_src(u)
            udst = model.edge_out_dst(u)
            osrc.append(usrc)
            odst.append(udst)
        osrc = torch.cat(osrc, 1)
        osrc = nn.Softmax()(osrc)
        osrc = torch.multinomial(osrc)
        odst = torch.cat(odst, 1)
        odst = nn.Softmax()(odst)
        odst = torch.multinomial(odst)
        otype = nn.Softmax()(otype)
        otype = torch.multinomial(otype)
        E[:, :, 0] = osrc[0, 0]
        E[:, :, 1] = odst[0, 0]
        E[:, :, 2] = otype[0, 0]
        edges.append((osrc.data[0, 0], odst.data[0, 0], otype.data[0, 0]))
        if edges[-1] == (0, 0, 0):
            break
    return vertices, edges

def preprocess(s):
    from rdkit import Chem
    m = Chem.MolFromSmiles(s)
    Chem.Kekulize(m)
    vertices = []
    for atom in m.GetAtoms():
        num = atom.GetAtomicNum()
        vertices.append(num)
    edges = []
    for bond in m.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        type = _get_type(bond)
        if begin > end:
            begin, end = end, begin
        edges.append((begin, end, type))
    #special characters
    vertices = [0] + vertices + [0]
    edges = [(0, 0, 0)] + edges + [(0, 0, 0)]
    return vertices, edges

def _get_type(bond):
    from rdkit import Chem
    for idx, val in Chem.BondType.values.items():
        if val == bond.GetBondType():
            return idx
    return 0

def deprocess(vertices, edges):
    from rdkit.Chem import EditableMol, MolFromSmiles, Atom, MolToSmiles, Bond, BondType, Mol
    m = EditableMol(Mol())
    for v in vertices:
        if v != 0:
            m.AddAtom(Atom(v))
    for begin, end, type in edges:
        if (begin, end) == (0, 0):
            continue
        if begin == end:
            continue
        if begin == len(vertices) - 1:
            continue
        if end == len(vertices) - 1:
            continue
        if begin > end:
            begin, end = end, begin
        if type == 0:
            continue
        if m.GetMol().GetBondBetweenAtoms(begin, end):
            continue
        m.AddBond(begin, end, BondType.values[type])
    s = MolToSmiles(m.GetMol())
    return s

def save(vertices, edges, out='out.png'):
    from rdkit.Chem import Draw, MolFromSmiles
    s = deprocess(vertices, edges)
    m = MolFromSmiles(s)
    if s == '':
        raise ValueError()
    if m:
        Draw.MolToFile(m, out, size=(800, 800))
        return s
    else:
        raise ValueError()

def pad(corpus):
    max_vertex_length = max(len(v) for v, e in corpus)
    max_edge_length = max(len(e) for v, e in corpus)
    corpus = [
        ( _pad_seq(v, max_vertex_length, zero=0), 
          _pad_seq(e, max_edge_length, zero=(0, 0, 0))
        )
        for v, e in corpus
    ]
    return corpus

def _pad_seq(seq, max_length, zero=(0,)):
    return seq + (max_length - len(seq)) * [zero]

def randomize(s):
    v, e = s
    v = v[1:-1]
    e = e[1:-1]
    perm = defaultdict(int)
    idx = np.arange(len(v))
    np.random.shuffle(idx)
    idx = idx.tolist()
    for i, j in enumerate(idx):
        perm[i] = j
    v = [v[j] for j in idx]
    e = [(perm[begin], perm[end], t) for begin, end, t in e]
    v = [0] + v + [0]
    e = [(0, 0, 0)] + e + [(0, 0, 0)]
    return v, e

def get_moving(moving, stats):
    for k, v in stats.items():
        prev = moving[k]
        new_ = v[-1]
        moving[k] = prev * 0.99 + new_ * 0.01
    return moving


def main():
    import pandas as pd
    ###
    if not os.path.exists('{{folder}}'):
        os.mkdir('{{folder}}')
    df = pd.read_csv('chembl22.csv')
    corpus = df['smiles'].values
    corpus = corpus[0:100000]
    corpus = list(map(preprocess, corpus))
    corpus = [(v, e) for v, e in corpus if len(v) < 100 and len(e) < 100]
    print(len(corpus))
    padding = False
    if padding:
        corpus = pad(corpus)
    max_vertex_length = max(len(v) for v,e in corpus)
    max_edge_length = max(len(e) for v, e in corpus)
    crit = nn.CrossEntropyLoss()
    if cuda: crit = crit.cuda()
    vocab_size = max(max(v) for v,e in corpus) + 1
    print('max vertices length : {}, max edges length : {}, vocab size : {}'.format(max_vertex_length, max_edge_length, vocab_size))
    hidden_size = {{'hidden_size'|choice(32, 64, 96, 128, 192, 256, 512, 800)}}
    output_size = vocab_size
    n_layers = 1
    batch_size = 1
    nb_epochs = 100
    emb_size = {{'emb_size'|choice(50, 100, 200, 300)}}
    n_edge_types = 22
    model = RNN(
        input_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size,  
        n_edge_types=n_edge_types, 
        max_length=max_vertex_length, 
        repr_size=emb_size)
    if cuda: model = model.cuda()

    algo = {{'algo'|choice(0, 1, 2)}}
    lr = {{'lr'|loguniform(-5, -1)}}
    if algo == 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif algo == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
    elif algo == 2:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    v, e = corpus[0]
    save(v, e, out='{{folder}}/true.png')
    j = 0
    k = 0
    gen_correct = 0.
    generated = []
    stats = defaultdict(list)
    with open('{{folder}}/generated', 'w'):
        pass
    moving = defaultdict(float)
    for e in range(nb_epochs):
        print('Epoch {}...'.format(e))
        for i in range(0, len(corpus), batch_size):
            model.zero_grad()
            t0 = time.time()

            corpus_cur = corpus[i:i + batch_size]
            #corpus_cur = list(map(randomize, corpus_cur))
            input_V = [v[0:-1] for v, _ in corpus_cur]
            target_V = [v[1:] for v, _ in corpus_cur]

            input_E = [e[0:-1] for _, e in corpus_cur]
            target_E = [e[1:] for _, e in corpus_cur]

            input_V = Variable(torch.LongTensor(input_V))
            input_E =  Variable(torch.LongTensor(input_E))

            target_V = Variable(torch.LongTensor(target_V))
            target_V = target_V.view(-1)

            target_E =  Variable(torch.LongTensor(target_E))
            target_E_src = target_E[:, :, 0].contiguous()
            target_E_src = target_E_src.view(-1)
            target_E_dst = target_E[:, :, 1].contiguous()
            target_E_dst = target_E_dst.view(-1)
            target_E_type = target_E[:, :, 2].contiguous()
            target_E_type = target_E_type.view(-1)

            if cuda: input_V = input_V.cuda()
            if cuda: input_E = input_E.cuda()
            output_V, output_E, output_E_type = model(input_V, input_E)

            # output_V      : nb_examples, vocab_size, nb_vertex_timesteps
            # output_E      : nb_examples, nb_vert_timesteps, 2, nb_edge_timesteps
            # output_E_type : nb_examples, nb_type_edges, nb_edge_timesteps

            output_V = output_V.transpose(1, 2).contiguous()
            output_V = output_V.view(output_V.size(0) * output_V.size(1), output_V.size(2))

            output_E_src = output_E[:, :, 0, :].transpose(1, 2).contiguous()
            output_E_src = output_E_src.view(output_E_src.size(0) * output_E_src.size(1), output_E_src.size(2))
            output_E_dst = output_E[:, :, 1, :].transpose(1, 2).contiguous()
            output_E_dst = output_E_dst.view(output_E_dst.size(0) * output_E_dst.size(1), output_E_dst.size(2))
            output_E_type = output_E_type.transpose(1, 2).contiguous()
            output_E_type = output_E_type.view(output_E_type.size(0) * output_E_type.size(1), output_E_type.size(2))

            if cuda: 
                target_E_src = target_E_src.cuda()
                target_E_dst = target_E_dst.cuda()
                target_E_type = target_E_type.cuda()
                target_V = target_V.cuda()

            loss = (
                    crit(output_V, target_V) + 
                    crit(output_E_src, target_E_src) + 
                    crit(output_E_dst, target_E_dst) +
                    crit(output_E_type, target_E_type))
            acc_V = acc(output_V, target_V)
            acc_E_src = acc(output_E_src, target_E_src)
            acc_E_dst = acc(output_E_dst, target_E_dst)
            acc_E_type = acc(output_E_type, target_E_type)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 2)
            optimizer.step()
            dt = (time.time() - t0)

            stats['loss'].append(loss.data[0])
            stats['acc_V'].append(acc_V.data[0])
            stats['acc_E_src'].append(acc_E_src.data[0])
            stats['acc_E_dst'].append(acc_E_dst.data[0])
            stats['acc_E_type'].append(acc_E_type.data[0])
            stats['dt'].append(dt)
            moving = get_moving(moving, stats)
            if j % 10 == 0:
                vert, edge = generate(
                    model, 
                    n_vertex_steps=max_vertex_length, 
                    n_edge_steps=max_edge_length)
                out = '{{folder}}/out_{:05d}.png'.format(k)
                try:
                    s = save(vert, edge, out=out)
                except Exception:
                    gen_correct = gen_correct * 0.99 + 0 * 0.01
                else:
                    with open('{{folder}}/generated', 'a') as fd:
                        fd.write(s + '\n')
                    gen_correct = gen_correct * 0.99 + 1 * 0.01
                    k += 1
            
            if j % 10000 == 0:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    lr /= 2.
                    param_group['lr'] = lr

            if j % 100 == 0:
                fmt = [
                    moving['loss'],
                    moving['acc_V'],
                    moving['acc_E_src'],
                    moving['acc_E_dst'],
                    moving['acc_E_type'],
                    moving['dt'],
                    gen_correct
                ]
                pd.DataFrame.from_dict(stats).to_csv('{{folder}}/stats.csv', index=False)
                torch.save(model.state_dict(), '{{folder}}/model.th')
                print('loss : {:.3f}, acc_V : {:.3f}, acc_E_src : {:.3f}, acc_E_dst : {:.3f}, acc_E_type : {:.3f}, time : {:.3f}, generated ratio: {:.3f}'.format(*fmt))
            j += 1

if __name__ == '__main__':
    main()
