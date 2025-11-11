"""Checkpointing utilities for data pipelines.

Provides a unified checkpointing system that handles:
- HuggingFace datasets (via state_dict/load_state_dict)
- Grain datasets and sources
- Regular Python objects (via pickle)
- Custom serializers via registry pattern
"""

from typing import Protocol, Any, Callable, TypeVar, runtime_checkable
import pickle
import inspect
from pathlib import Path

try:
    import datasets
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

try:
    import grain.python as grain
    HAS_GRAIN = True
except ImportError:
    HAS_GRAIN = False


@runtime_checkable
class Stateful(Protocol):
    """Protocol for objects that support state_dict/load_state_dict pattern."""

    def state_dict(self) -> dict:
        """Return serializable state dictionary."""
        ...

    def load_state_dict(self, state: dict) -> None:
        """Restore from state dictionary."""
        ...


T = TypeVar('T')


class CheckpointRegistry:
    """Registry for custom serialization strategies.

    Usage:
        # Register custom serializer
        checkpoint.register(
            type_=MyClass,
            serialize_fn=lambda obj: obj.to_dict(),
            deserialize_fn=lambda data: MyClass.from_dict(data),
            name='my_class'
        )

        # Save/load
        state = checkpoint.save(my_obj)
        restored = checkpoint.load(state)
    """

    def __init__(self):
        self._serializers: dict[type, tuple[Callable, str]] = {}
        self._deserializers: dict[str, Callable] = {}

    def register(
        self,
        type_: type,
        serialize_fn: Callable[[Any], Any],
        deserialize_fn: Callable[[Any], Any],
        name: str = None
    ):
        """Register custom serializer for a type.

        Args:
            type_: The type to register serializer for
            serialize_fn: Function that takes object and returns serializable data
            deserialize_fn: Function that takes data and returns object
            name: Optional name for the serializer (defaults to type's qualified name)
        """
        name = name or f"{type_.__module__}.{type_.__name__}"
        self._serializers[type_] = (serialize_fn, name)
        self._deserializers[name] = deserialize_fn

    def save(self, obj: Any) -> dict:
        """Serialize object to dictionary.

        Tries in order:
        1. Registered custom serializer
        2. state_dict() protocol
        3. HuggingFace dataset state_dict
        4. Pickle (fallback)

        Args:
            obj: Object to serialize

        Returns:
            Dictionary with '__type__' and '__data__' keys
        """
        # Check for registered type
        for type_, (serializer, name) in self._serializers.items():
            if isinstance(obj, type_):
                return {
                    '__type__': name,
                    '__data__': serializer(obj)
                }

        # Check for state_dict protocol
        if isinstance(obj, Stateful):
            return {
                '__type__': 'stateful',
                '__class__': f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                '__data__': obj.state_dict()
            }

        # Check for HuggingFace dataset
        if HAS_HF_DATASETS and hasattr(obj, 'state_dict') and callable(obj.state_dict):
            return {
                '__type__': 'hf_dataset',
                '__class__': f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                '__data__': obj.state_dict()
            }

        # Check for generator/iterator
        if inspect.isgenerator(obj) or inspect.isgeneratorfunction(obj):
            raise ValueError(
                f"Cannot directly checkpoint generator {obj}. "
                f"Wrap in a CheckpointableIterator or stateful class."
            )

        # Default: pickle
        return {
            '__type__': 'pickle',
            '__data__': pickle.dumps(obj)
        }

    def load(self, data: dict) -> Any:
        """Deserialize object from dictionary.

        Args:
            data: Dictionary from save() method

        Returns:
            Deserialized object
        """
        type_name = data['__type__']

        if type_name == 'pickle':
            return pickle.loads(data['__data__'])

        if type_name in ('stateful', 'hf_dataset'):
            # For stateful objects, we need the class to exist
            # User must ensure the class is importable
            module_name, class_name = data['__class__'].rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)

            # Instantiate and load state
            obj = object.__new__(cls)
            if hasattr(obj, 'load_state_dict'):
                obj.load_state_dict(data['__data__'])
            else:
                # Fallback: try __init__ with no args then set __dict__
                obj.__init__()
                obj.__dict__.update(data['__data__'])
            return obj

        # Custom deserializer
        if type_name in self._deserializers:
            return self._deserializers[type_name](data['__data__'])

        raise ValueError(f"Unknown type: {type_name}")

    def save_to_file(self, obj: Any, path: str | Path):
        """Save object to file."""
        import json
        state = self.save(obj)
        with open(path, 'w') as f:
            json.dump(state, f)

    def load_from_file(self, path: str | Path) -> Any:
        """Load object from file."""
        import json
        with open(path, 'r') as f:
            state = json.load(f)
        return self.load(state)


# Global registry instance
checkpoint = CheckpointRegistry()


class CheckpointableIterator:
    """Wrapper that makes iterators checkpointable.

    Usage:
        def create_iterator(dataset):
            return iter(dataset)

        ckpt_iter = CheckpointableIterator(dataset, create_iterator)

        # Use it
        for item in ckpt_iter:
            process(item)

        # Checkpoint
        state = ckpt_iter.state_dict()

        # Restore
        ckpt_iter = CheckpointableIterator.from_state_dict(
            state, dataset, create_iterator
        )
    """

    def __init__(self, source: Any, create_fn: Callable[[Any], Any]):
        """
        Args:
            source: The underlying iterable (dataset, list, etc.)
            create_fn: Function to recreate iterator from source
        """
        self.source = source
        self.create_fn = create_fn
        self.iterator = create_fn(source)
        self.step = 0

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.iterator)
        self.step += 1
        return item

    def state_dict(self) -> dict:
        """Save current position and source state."""
        state = {'step': self.step}

        # Try to save source state
        if hasattr(self.source, 'state_dict'):
            state['source_state'] = self.source.state_dict()
        else:
            # Pickle the source if small enough
            try:
                state['source'] = pickle.dumps(self.source)
            except (pickle.PicklingError, TypeError):
                # Source not picklable - user must recreate manually
                state['source'] = None

        return state

    @classmethod
    def from_state_dict(
        cls,
        state: dict,
        source: Any,
        create_fn: Callable[[Any], Any]
    ):
        """Restore from checkpoint.

        Args:
            state: State dictionary from state_dict()
            source: Source object (must be provided by user if not serializable)
            create_fn: Iterator creation function
        """
        # Restore source if it was serialized
        if 'source_state' in state and hasattr(source, 'load_state_dict'):
            source.load_state_dict(state['source_state'])
        elif state.get('source') is not None:
            source = pickle.loads(state['source'])

        # Create new instance
        obj = cls(source, create_fn)

        # Fast-forward to saved step
        for _ in range(state['step']):
            next(obj.iterator)
        obj.step = state['step']

        return obj


if HAS_GRAIN:
    class CheckpointableStreamSource(grain.RandomAccessDataSource):
        """Wrap streaming source as Grain random-access source with checkpointing.

        This follows MaxText's pattern of faking a RandomAccessDataSource
        with an iterator, but adds proper checkpointing support.

        Usage:
            hf_dataset = datasets.load_dataset(..., streaming=True)
            source = CheckpointableStreamSource(hf_dataset)

            # Build Grain pipeline
            dataset = grain.MapDataset.source(source)
            dataset = dataset.map(tokenize)
            # ...

            # Checkpoint
            state = {
                'source': source.state_dict(),
                # Save other pipeline state...
            }

            # Restore
            source = CheckpointableStreamSource.from_state_dict(
                state['source'],
                lambda: datasets.load_dataset(..., streaming=True)
            )
        """

        def __init__(self, source: Any):
            """
            Args:
                source: Underlying streaming source (HF IterableDataset, etc.)
            """
            self.source = source
            self._iterator = None
            self._current_idx = 0

        def __len__(self):
            """Return fake length for streaming (required by Grain)."""
            return 10_000_000_000

        def __getitem__(self, idx):
            """Return next item from stream.

            Note: idx is ignored - this always returns the next item.
            This is intentional for streaming sources.
            """
            if self._iterator is None:
                self._iterator = iter(self.source)

            try:
                item = next(self._iterator)
                self._current_idx += 1
                return item
            except StopIteration:
                # Loop back to start
                self._iterator = iter(self.source)
                self._current_idx = 0
                return next(self._iterator)

        def state_dict(self) -> dict:
            """Checkpoint current position."""
            state = {'current_idx': self._current_idx}

            # Try to save source state
            if hasattr(self.source, 'state_dict'):
                state['source_state'] = self.source.state_dict()

            return state

        @classmethod
        def from_state_dict(
            cls,
            state: dict,
            source_factory: Callable[[], Any]
        ):
            """Restore from checkpoint.

            Args:
                state: State dictionary from state_dict()
                source_factory: Function to recreate the source (e.g., lambda: load_dataset(...))
            """
            # Recreate source
            source = source_factory()

            # Restore source state if available
            if 'source_state' in state and hasattr(source, 'load_state_dict'):
                source.load_state_dict(state['source_state'])

            # Create instance
            obj = cls(source)
            obj._current_idx = state['current_idx']

            # Note: We don't fast-forward here because HF's load_state_dict
            # already handles the resumption point

            return obj
