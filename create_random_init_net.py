import numpy as np

with open('random_init.net', 'w') as g:
    with open('eval.net', 'r') as f:
        num_layers = int(f.readline())
        g.write("%d\n" % num_layers)
        for layer in xrange(num_layers):               
            for k in xrange(3):
                input_size = int(f.readline())
                output_size = int(f.readline())
                g.write("%d\n" % input_size)
                g.write("%d\n" % output_size)
                if k != 1:
                    if layer == num_layers - 1:
                        r = np.sqrt(6.0 / (input_size + output_size))
                        random_mat = np.random.rand(input_size, output_size)
                        random_mat *= 2 * r
                        random_mat -= r
                    else:
                        random_mat = np.random.randn(input_size + output_size + 1, 4 * output_size) / np.sqrt(input_size + output_size)
                    for i in xrange(input_size):
                        row_str = ""
                        for j in xrange(output_size):
                            row_str += "%f " % random_mat[i,j]
                        g.write("%s\n" % row_str[:-1])
                    for i in xrange(input_size):
                        f.readline()
                else:
                    for i in xrange(input_size):
                        line = f.readline()
                        g.write(line)
        rest = f.read()
        g.write(rest)

