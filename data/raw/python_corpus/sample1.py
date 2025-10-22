class B:
    def bar(self):
        pass

class A(B):
    def foo(self):
        self.bar()


def util():
    from math import sqrt
    return sqrt(4)
