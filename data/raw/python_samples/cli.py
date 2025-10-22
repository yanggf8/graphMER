import sys
from typing import Callable, Any, Optional, List, Dict
from functools import wraps

class Context:
    def __init__(self):
        self.params = {}
        self.parent = None
    
    def find_root(self):
        ctx = self
        while ctx.parent:
            ctx = ctx.parent
        return ctx

class Command:
    def __init__(self, name: str, callback: Callable, params: List = None):
        self.name = name
        self.callback = callback
        self.params = params or []
    
    def invoke(self, ctx: Context):
        return self.callback(**ctx.params)

class Group(Command):
    def __init__(self, name: str = None):
        super().__init__(name, self._group_callback)
        self.commands = {}
    
    def _group_callback(self, **kwargs):
        pass
    
    def add_command(self, cmd: Command):
        self.commands[cmd.name] = cmd
    
    def command(self, name: str = None):
        def decorator(f):
            cmd_name = name or f.__name__
            cmd = Command(cmd_name, f)
            self.add_command(cmd)
            return f
        return decorator

def command(name: str = None):
    def decorator(f):
        cmd_name = name or f.__name__
        return Command(cmd_name, f)
    return decorator

def option(name: str, default: Any = None, help: str = None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    return decorator

def argument(name: str, required: bool = True):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return wrapper
    return decorator

class CLI:
    def __init__(self):
        self.group = Group()
    
    def parse_args(self, args: List[str]) -> Context:
        ctx = Context()
        # Simple parsing logic
        for i, arg in enumerate(args):
            if arg.startswith('--'):
                key = arg[2:]
                value = args[i + 1] if i + 1 < len(args) else True
                ctx.params[key] = value
        return ctx
    
    def run(self, args: List[str] = None):
        if args is None:
            args = sys.argv[1:]
        
        ctx = self.parse_args(args)
        
        if args and args[0] in self.group.commands:
            cmd = self.group.commands[args[0]]
            return cmd.invoke(ctx)
        else:
            self.show_help()
    
    def show_help(self):
        print("Usage: cli [OPTIONS] COMMAND [ARGS]...")
        print("\nCommands:")
        for name, cmd in self.group.commands.items():
            print(f"  {name}")

# Example usage
cli = CLI()

@cli.group.command()
@option('--count', default=1, help='Number of greetings')
@argument('name')
def hello(name: str, count: int):
    for _ in range(count):
        print(f"Hello {name}!")

@cli.group.command()
@option('--format', default='json', help='Output format')
def status(format: str):
    if format == 'json':
        print('{"status": "ok"}')
    else:
        print("Status: OK")
