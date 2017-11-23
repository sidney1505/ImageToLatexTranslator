import sys, os, re, shutil, argparse, logging, code
sys.path.insert(0, '%s'%os.path.join(os.path.dirname(__file__), '../utils/'))
from runner import run
from image_utils import *
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 


TIMEOUT = 10

# replace \pmatrix with \begin{pmatrix}\end{pmatrix}
# replace \matrix with \begin{matrix}\end{matrix}
template = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\begin{displaymath}
%s
\end{displaymath}
\end{document}
"""

def render_output(model_dir, result_path, output_dir):
    assert os.path.exists(result_path), result_path
    replace = True
    num_threads = 1
    # create directories
    pred_dir = os.path.join(output_dir, 'images_pred')
    gold_dir = os.path.join(output_dir, 'images_gold')
    for dirname in [pred_dir, gold_dir]:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    lines = []
    with open(result_path) as fin:
        for line in fin:
            try:
                img_path, label_gold, label_pred, _, _ = line.strip().split('\t')
                lines.append((img_path, label_gold, os.path.join(gold_dir, img_path), \
                    replace))
                lines.append((img_path, label_pred, os.path.join(pred_dir, img_path), \
                    replace))
            except Exception, e:
                import code
                code.interact(local=dict(globals(), **locals()))
            else:
                pass
            finally:
                pass
    # print('Creating pool with %d threads'%num_threads)
    # pool = ThreadPool(num_threads)
    # print('Jobs running...')
    # results = pool.map(main_parallel, lines)
    # pool.close() 
    # pool.join()
    i = 0
    pre_dir =  model_dir + '/tmp'
    pre_dir = pre_dir.replace(' ','_')
    pre_dir = pre_dir.replace(':','_')
    pre_dir = pre_dir.replace('.','_')
    if not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    for line in lines:
        img_path, l, output_path, replace = line
        pre_name = pre_dir +  '/' + output_path[-14:].replace('/', '_').replace('.','_')
        l = l.strip()
        l = l.replace(r'\pmatrix', r'\mypmatrix')
        l = l.replace(r'\matrix', r'\mymatrix')
        # remove leading comments
        l = l.strip('%')
        if len(l) == 0:
            l = '\\hspace{1cm}'
        # \hspace {1 . 5 cm} -> \hspace {1.5cm}
        for space in ["hspace", "vspace"]:
            match = re.finditer(space + " {(.*?)}", l)
            if match:
                new_l = ""
                last = 0
                for m in match:
                    new_l = new_l + l[last:m.start(1)] + m.group(1).replace(" ", "")
                    last = m.end(1)
                new_l = new_l + l[last:]
                l = new_l    
        if replace or (not os.path.exists(output_path)):
            tex_filename = pre_name+'.tex'
            log_filename = pre_name+'.log'
            aux_filename = pre_name+'.aux'
            pdf_filename = pre_name+'.pdf'
            png_filename = pre_name+'.png'
            import code
            with open(tex_filename, "w") as w: 
                print >> w, (template%l)
            run("pdflatex -interaction=nonstopmode -output-directory="+pre_dir+" "+\
                tex_filename +"  >/dev/null", TIMEOUT)
            if not os.path.exists(log_filename):
                code.interact(local=dict(globals(), **locals()))
            #code.interact(local=dict(globals(), **locals()))
            os.remove(log_filename)
            os.remove(aux_filename)
            if not os.path.exists(pdf_filename):
                print('cannot compile ' + img_path + ' to ' + output_path)
            else:
                os.system("convert -density 200 -quality 100 %s %s"%(pdf_filename, png_filename))
                os.remove(pdf_filename)
                if os.path.exists(png_filename):
                    crop_image(png_filename, output_path)
                    os.remove(png_filename)
            os.remove(tex_filename)
        if i % 10 == 0:
            print(i)
        i = i + 1

def main_parallel(line):
    img_path, l, output_path, replace = line
    pre_name = output_path[-2:].replace('/', '_').replace('.','_')
    l = l.strip()
    l = l.replace(r'\pmatrix', r'\mypmatrix')
    l = l.replace(r'\matrix', r'\mymatrix')
    # remove leading comments
    l = l.strip('%')
    if len(l) == 0:
        l = '\\hspace{1cm}'
    # \hspace {1 . 5 cm} -> \hspace {1.5cm}
    for space in ["hspace", "vspace"]:
        match = re.finditer(space + " {(.*?)}", l)
        if match:
            new_l = ""
            last = 0
            for m in match:
                new_l = new_l + l[last:m.start(1)] + m.group(1).replace(" ", "")
                last = m.end(1)
            new_l = new_l + l[last:]
            l = new_l    
    if replace or (not os.path.exists(output_path)):
        tex_filename = pre_name+'.tex'
        log_filename = pre_name+'.log'
        aux_filename = pre_name+'.aux'
        with open(tex_filename, "w") as w: 
            print >> w, (template%l)
        run("pdflatex -interaction=nonstopmode %s  >/dev/null"%tex_filename, TIMEOUT)
        import code
        #code.interact(local=dict(globals(), **locals()))
        os.remove(log_filename)
        os.remove(aux_filename)
        pdf_filename = tex_filename[:-4]+'.pdf'
        png_filename = tex_filename[:-4]+'.png'
        if not os.path.exists(pdf_filename):
            print('cannot compile ' + img_path + ' to ' + output_path)
        else:
            os.system("convert -density 200 -quality 100 %s %s"%(pdf_filename, png_filename))
            os.remove(pdf_filename)
            if os.path.exists(png_filename):
                crop_image(png_filename, output_path)
                os.remove(png_filename)
        os.remove(tex_filename)
    global i
    if i % 10 == 0:
        print(i)
    i = i + 1

def main():
    print('enter main method')
    result_path = '/cvhci/data/docs/math_expr/printed/im2latex-100k/models/' + \
        'final_models_torch/old/wysiwyg/results/results.txt'
    image_dir = '/cvhci/data/docs/math_expr/printed/im2latex-100k/models/' + \
        'final_models_torch/old/wysiwyg/rendered'
    render_output(result_path, image_dir)
    #code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':
    main()