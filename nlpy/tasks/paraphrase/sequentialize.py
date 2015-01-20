from nlpy.syntax.cfg import StanfordCFGParser, BatchStanfordCFGParser
from nlpy.basic import DefaultTokenizer
import sys, StringIO

class CFGSequencer(object):

    def __init__(self, tree):
        self.tree = tree
        pass

    def _sequence(self):
        """
        Suppose there are two registers: c1, c2.
        Make sequence to combine all nodes in the tree, output to c2.
        """
        # Depth-first iteration
        stack = [self.tree]
        register_used = [False] * 20
        processed_nodes = set()
        output_map = dict()
        while stack:
            node = stack[-1]
            tag, content = node
            # Only consider non-terminal nodes
            if type(content) == list:
                if repr(node) not in processed_nodes:
                    processed_nodes.add(repr(node))
                    non_terminals = []
                    for child in content:
                        if type(child[1]) == list:
                            non_terminals.append(child)
                    stack.extend(non_terminals)

                else:
                    my_output = register_used.index(False)
                    output_map[repr(node)] = my_output
                    if len(content) == 1:
                        # One child
                        first_child = content[0]
                        if type(first_child[1]) == list:
                            # Copy c2 to c2, so do nothing
                            output_map[repr(node)] = output_map[repr(first_child)]
                        else:
                            yield -1, first_child[1], my_output
                    else:
                        # Multiple children
                        right_node = content[-1]
                        right = output_map[repr(right_node)] if type(right_node[1]) == list else right_node[1]
                        if type(right) == int:
                            register_used[right] = False
                        for i in reversed(range(len(content) - 1)):
                            left_node = content[i]
                            left = output_map[repr(left_node)] if type(left_node[1]) == list else left_node[1]
                            yield left, right, my_output
                            right = my_output
                            if type(left) == int:
                                register_used[left] = False
                    register_used[my_output] = True
                    stack.pop()

    def _optimize(self, seq):
        register_map = dict()
        for left, right, output in seq:
            if left == -1:
                register_map[output] = right
            else:
                if left in register_map:
                    left_copy = register_map[left]
                    del register_map[left]
                    left = left_copy
                if right in register_map:
                    right_copy = register_map[right]
                    del register_map[right]
                    right = right_copy
                yield left, right, output


    def _recude_number(self, seq):
        register_used = [False] * 20
        rewrite_rules = {}
        for left, right, output in seq:
            if type(left) == int:
                if left in rewrite_rules:
                    new_left = rewrite_rules[left]
                    del rewrite_rules[left]
                    left = new_left
                register_used[left] = False
            else:
                left = "[%s]" % left
            if type(right) == int:
                if right in rewrite_rules:
                    new_right = rewrite_rules[right]
                    del rewrite_rules[right]
                    right = new_right
                register_used[right] = False
            else:
                right = "[%s]" % right
            new_output = register_used.index(False)
            if new_output != output:
                rewrite_rules[output] = new_output
                output = new_output
            register_used[output] = True
            yield left, right, output

    def __iter__(self):
        return self._recude_number(self._optimize(self._sequence()))



if __name__ == '__main__':

    tokenizer = DefaultTokenizer()

    sent_list = [x.strip() for x in sys.stdin.xreadlines()]
    tok_list = [tokenizer.tokenize(x) for x in sent_list]

    if len(sys.argv) == 2:
        parser = BatchStanfordCFGParser()
        parser.load_output(tok_list, sys.argv[1])
    else:
        parser = StanfordCFGParser()


    testcase = "A discouraging outlook from General Electric Co. sent its share down 81 cents (U.S.) or 2.7 per cent to $29.32."
    #sys.stderr = StringIO.StringIO()
    reload(sys)
    sys.setdefaultencoding("utf-8")

    # tree = parser.parse(tokenizer.tokenize(testcase))
    # print tree
    # print list(CFGSequencer(tree))
    # raise SystemExit

    for l in sent_list:
        tree = parser.parse(tokenizer.tokenize(l.strip()))
        if not tree:
            print >> sys.stderr, "skip:", l
            continue
        seq = []
        try:
            seq = list(CFGSequencer(tree))
        except:
            print >> sys.stderr, "skip:", l
            continue
        for x in seq:
            print "_".join(map(str, x)),
        print ""
