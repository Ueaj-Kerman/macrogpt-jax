import re
from flax import nnx

from flax.core import FrozenDict
from ueaj.model.ueajsum.config import (
    UeajsumConfig, ArgumentConfig, ParamConfig
)

SPECIAL_SYMBOLS = {
	"*": nnx.Param,
	"&": nnx.LoRAParam,
}

def _make_default(shape: str, group: nnx.Variable | None):
	if group is None:
		return ArgumentConfig(shape=shape)
	else:
		return ParamConfig(shape=shape, group=group)

def parse(expr, no_instantiation: bool = False, *args, **kwargs) -> UeajsumConfig:
	if "->" in expr:
		lhs, rhs = expr.split("->")
	else:
		lhs, rhs = expr, ""

	rhs = ArgumentConfig(shape=rhs.strip())

	sums = lhs.split("+")

	kwargs = dict(kwargs)
	args = list(args)

	c = 0
	access_specs = []
	for term in sums:
		access_spec = []

		for arg in term.split(","):
			arg = arg.strip()
			group = None
			# parse group type
			for symbol, cls in SPECIAL_SYMBOLS.items():
				if arg.startswith(symbol):
					group = cls
					arg = arg[len(symbol):]
					break

			if "=" in arg: # named initialization
				if no_instantiation and group:
					raise ValueError(f"Instantiation not allowed in execution!")

				name, shape = arg.split("=")
				if name in kwargs:
					raise ValueError(f"Duplicate argument assignment for: {name} in '{expr}'")

				kwargs[name] = _make_default(shape, group)

				# kwarg access
				access_spec.append(name)
			elif m := re.match(r"\[(.*)]", arg): # indexed arg
				idx = m.groups()
				try:
					idx = int(idx[0])
				except ValueError as e:
					raise ValueError(f"Invalid index '{idx}' in '{expr}'") from e

				access_spec.append(idx)
			elif m := re.match(r"\{(.*)}", arg): # named arg
				idx = m.groups()
				assert len(idx) == 1
				access_spec.append(idx[0])
			else: # positional initialization
				if no_instantiation and group:
					raise ValueError(f"Instantiation not allowed!")

				shape = arg
				while c < len(args) and args[c] is not None:
					c += 1
				if c >= len(args):
					args.append(_make_default(shape, group))
				else:
					args[c] = _make_default(shape, group)

				access_spec.append(c)

		access_specs.append(tuple(access_spec))

	# validate access specs after in case out of order initializations
	for access_spec in access_specs:
		for spec in access_spec:
			if spec in kwargs:
				continue
			if isinstance(spec, str):
				raise ValueError(f"Unknown argument: {spec} in '{expr}'")
			elif isinstance(spec, int):
				if spec >= len(args) or args[spec] is None:
					raise ValueError(f"Index out of bounds: {spec} in '{expr}'")

	return UeajsumConfig(
		no_instantiation=no_instantiation,
		arg_configs=tuple(args),
		kwarg_configs=FrozenDict(kwargs),
		result_config=rhs,
		sums=tuple(access_specs)
	)

if __name__ == "__main__":
	config = parse("bnd,*dhik->bnhik")
	print(config.muP_param_var({"d": 3, "h": 4, "i": 4, "k": 6}))