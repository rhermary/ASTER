"""Implementation of the ASTER model."""

from abc import abstractmethod, ABCMeta
from enum import Enum, auto, Flag
from typing import Any, Tuple, TypeVar, Type, List, Iterable, Iterator, Optional
import os
from contextlib import contextmanager
# from pytorch_lightning.utilities.rank_zero import rank_zero_only
# import mlflow
import itertools

from transformers import PreTrainedModel

import torch
import math
from torch import Tensor
from torch.nn import Module

from torch.autograd import Function
# from datasets.base import BaseDataModule
from enum import Enum, auto
from typing import Any

from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor
from torch.nn import Parameter
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
	BaseModelOutputWithPastAndCrossAttentions,
)

from transformers import GPT2Model

from torch import Tensor
from torch.nn import (
	TransformerEncoderLayer, TransformerEncoder, Dropout, Linear, ModuleList,
	Sequential, LayerNorm,
	TransformerEncoderLayer,
	TransformerEncoder,
	TransformerDecoderLayer,
	TransformerDecoder,
	Parameter, SiLU, BatchNorm1d,
	BCEWithLogitsLoss as TorchBCEWithLogitsLoss,
	MSELoss as TorchMSELoss,
)


@contextmanager
def ensure_hf_cache(cache_dir: str) -> Iterator[None]:
	old = {k: os.environ.get(k) for k in ("HF_HOME", "TRANSFORMERS_CACHE")}
	# os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"] = cache_dir

	try:
		yield
	finally:
		for k, v in old.items():
			if v is None:
				os.environ.pop(k, None)
			else:
				os.environ[k] = v

# class Base(metaclass=ABCMeta):
#     """Base class for all classes."""

#     @classmethod
#     @rank_zero_only
#     def log_param(cls: Type["B"], key: str, value: Any) -> None:
#         """Wrapper around `mlflow.log_param` ensuring a run is started.

#         Args:
#             cls (Type[B]): Current object class.
#             key (str): Key for the value that will be logged.
#             value (Any): Value to be logged.
#         """
#         if mlflow.active_run() is not None:
#             mlflow.log_param(key, value)


# Creating this type variable allows better static sub-typing/type resolution
# in the subclasses' methods.
# B = TypeVar("B", bound=Base)

class PLUMESubModule(Module, metaclass=ABCMeta):
	"""Base class for ASTER's perturbator and classifier."""

	def __init__(self) -> None:
		super().__init__()

		self.loss: Tensor
		self.register_buffer("loss", torch.tensor(0.0))

	@abstractmethod
	def forward(  # pylint: disable=missing-param-doc
		self, inputs: Tensor, *args: Any, **kwargs: Any
	) -> Tensor:
		"""Forward pass implementation."""

class PoolingType(Enum):
	MEAN = auto()
	CLS = auto()

class Transformer(PLUMESubModule):
	def __init__(
		self,
		depth: int,
		emb_size: int,
		num_heads: int = 6,
		dropout: float = 0.1,
		pooling: PoolingType = PoolingType.MEAN,
		num_class_tokens: int = 1,
		positional_encoding: bool = False,
	):
		super().__init__()
		self.dropout_ = dropout
		self.pooling = pooling
		self.num_heads = num_heads
		self.depth = depth
		self.num_class_tokens = num_class_tokens
		self.embedding_size: int = emb_size
		self.positional_encoding_ = positional_encoding

		self.dropout = Dropout(self.dropout_)

		self.positional_encoding: PositionalEncoding | None = None
		if self.positional_encoding_:
			self.positional_encoding = PositionalEncoding(self.embedding_size)

		encoder_layer = TransformerEncoderLayer(
			d_model=self.embedding_size,
			nhead=self.num_heads,
			dropout=self.dropout_,
			batch_first=True,
		)
		self.encoder = TransformerEncoder(encoder_layer, num_layers=self.depth)
		self.classifier = Linear(
			self.embedding_size * self.num_class_tokens,
		   1,
		)

	def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
		# inputs [B*2, W, D], exp: [128, 4, 768] 
		if self.positional_encoding:
			inputs = self.positional_encoding(inputs)

		encoded: Tensor = self.encoder(inputs)

		if self.pooling is PoolingType.MEAN:
			pooled = encoded.mean(dim=1)
		else:
			pooled = encoded[:, :self.num_class_tokens, :]
			pooled = pooled.reshape(pooled.size(0), -1)
		
		pooled = self.dropout(pooled)
		logits: Tensor = self.classifier(pooled)
		
		return logits.squeeze(1)

class Perturbator(PLUMESubModule, metaclass=ABCMeta):
	"""ASTER perturbator."""

class PositionalEncoding(Module):
	def __init__(self, embedding_size, max_len: int = 5000, only_positions: bool = False):
		super().__init__()

		self.only_positions = only_positions

		pe = torch.zeros(max_len, embedding_size)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		self.register_buffer("pe", pe.unsqueeze(0))

		self.pe: Tensor

	def forward(self, inputs: Tensor) -> Tensor:
		if self.only_positions:
			return self.pe[:, :inputs.size(1)].repeat(inputs.size(0), 1, 1)

		return inputs + self.pe[:, :inputs.size(1)]

class BaseLoss(metaclass=ABCMeta):
	"""Base class for the losses."""

	LOSS = "loss"

	def __init__(self, coefficient: float = 1.0) -> None:
		"""Initialise the loss.

		Args:
			coefficient (float, optional): Coefficient for the loss. Defaults
				to 1.0.
		"""
		self.coefficient = coefficient

class KLDivergence(BaseLoss):
	"""Kullback-Leibler divergence used in VAEs."""

	LOSS = "kl_divergence"

	def __call__(self, set1: Tensor, set2: Tensor, **kwargs: Any) -> Tensor:
		loss = (
			0.5
			* torch.sum(torch.exp(set2) + torch.pow(set1, 2) - 1.0 - set2)
			/ set1.size(0)
		)

		# self.log(loss)

		return self.coefficient * loss



class MSELoss(BaseLoss):
	"""Mean Square Error."""

	def __init__(self, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)

		self._loss = TorchMSELoss()

	def __call__(self, set1: Tensor, set2: Tensor, **kwargs: Any) -> Tensor:
		loss = self._loss(set1, set2)

		return self.coefficient * loss

class ReSamplingStrategy(Module, metaclass=ABCMeta):
	@abstractmethod
	def forward(self, samples: Tensor) -> Tensor:
		pass

class NoReSampling(ReSamplingStrategy):
	def forward(self, samples: Tensor) -> Tensor:
		return samples.detach()

class LogNormal(ReSamplingStrategy):
	def forward(self, samples: Tensor) -> Tensor:
		samples = samples.detach()

		return samples.sign() * samples.abs().exp()


class _Quantifier(metaclass=ABCMeta):
	""""""

class CoefficientScheduling(metaclass=ABCMeta):
	def __init__(self) -> None:
		self.last_coefficient = None

	@abstractmethod
	def __call__(self, current_step: int, max_steps: int, coefficient: float) -> float:
		pass

class FixedCoefficient(CoefficientScheduling):
	def __call__(self, current_step: int, max_steps: int, coefficient: float) -> float:
		self.last_coefficient = coefficient

		return coefficient

class _ScaledQuantifier(_Quantifier, metaclass=ABCMeta):
	def __init__(self, coefficient: float = 1.0, coefficient_scheduling: CoefficientScheduling | None = None) -> None:
		self.coefficient_ = coefficient
		self.coefficient_scheduling = coefficient_scheduling or FixedCoefficient()

	@property
	def coefficient(self) -> float:
		return self.coefficient_scheduling(
			None, None,
			self.coefficient_,
		)

class BaseDispersion(_ScaledQuantifier, metaclass=ABCMeta):
	"""Base dispersion class."""

	@abstractmethod
	def __call__(self, *, batch: Tensor) -> Tensor:
		"""Calculate the dispersion between each sample in the batch.

		Returns:
			Tensor: The dispersion value.
		"""

class EDTransformerVAE(Perturbator):
	class Projections(Flag):
		HIDDEN_TO_HIDDEN = auto()
		HIDDEN_TO_EMBEDDING = auto()
		OUTPUT = auto()
		NONE = auto()

	class PositionalEncoding_(Flag):
		SOURCE_LEVEL = auto()
		TARGET_LEVEL = auto()
		TARGET_LEVEL_POSONLY = auto()
		NONE = auto()

	class SamplingLevel(Enum):
		MEAN = auto()
		ALL = auto()

	class SecondaryTask(Enum):
		# NEXT_TOKEN = auto()
		# FORECASTING = auto()
		RECONSTRUCTION = auto()

	def __init__(
		self,
		parameters_loss: KLDivergence,
		reconstruction_loss: MSELoss,
		resampling_strategy: ReSamplingStrategy,
		depth: int,
		num_heads: int = 6,
		feature_size: int = 768,
		# hidden_size: int | None = None,
		# projections: Projections = Projections.NONE,
		positional_encoding: PositionalEncoding_ = PositionalEncoding_.NONE,
		secondary_task: SecondaryTask = SecondaryTask.RECONSTRUCTION,
		dropout: float = 0.1,
		sampling_level: SamplingLevel = SamplingLevel.MEAN,
		sampling_constraint: BaseDispersion | None = None,
		outputs_constraint: BaseDispersion | None = None,
		residual: bool = False,
		forecasting_length: int | None = None,
	) -> None:
		super().__init__()

		self.depth = depth
		self.num_heads = num_heads
		self.parameters_loss = parameters_loss
		self.reconstruction_loss = reconstruction_loss
		# self.hidden_size = hidden_size
		self.dropout = dropout
		# self.projections = projections
		self.positional_encoding_ = positional_encoding
		self.sampling_level = sampling_level
		self.sampling_constraint = sampling_constraint
		self.outputs_constraint = outputs_constraint
		self.residual = residual
		self.forecasting_length = forecasting_length
		self.secondary_task = secondary_task
		self.resampling_strategy = resampling_strategy

		self.positional_encoding: ModuleList = ModuleList([None, None])

		# self.hidden_projection: Sequential | None = None
		# self.output_projection: Linear | None = None

		self.setup_with_provider_(feature_size=feature_size)

	def setup_with_provider_(self, feature_size: int) -> None:
		# hidden_size = self.hidden_size or feature_size
		hidden_size = feature_size

		if self.PositionalEncoding_.SOURCE_LEVEL in self.positional_encoding_:
			self.positional_encoding[0] = PositionalEncoding(feature_size)

		encoder_layer = TransformerEncoderLayer(
			feature_size, self.num_heads, batch_first=True, dropout=self.dropout
		)
		self.encoder = TransformerEncoder(encoder_layer, num_layers=self.depth)

		self.encoder_mu = Linear(feature_size, hidden_size)
		self.encoder_sigma = Linear(feature_size, hidden_size)
		
		# if self.Projections.HIDDEN_TO_EMBEDDING in self.projections:
		#     self.hidden_projection = Sequential(
		#         Linear(hidden_size, feature_size),
		#         LayerNorm(feature_size),
		#         Dropout(self.dropout),
		#     )
		#     hidden_size = feature_size
		# elif self.Projections.HIDDEN_TO_HIDDEN in self.projections:
		#     self.hidden_projection = Sequential(
		#         Linear(hidden_size, hidden_size),
		#         LayerNorm(hidden_size),
		#         Dropout(self.dropout),
		#     )

		if self.PositionalEncoding_.TARGET_LEVEL in self.positional_encoding_:
			self.positional_encoding[1] = PositionalEncoding(hidden_size)
		elif self.PositionalEncoding_.TARGET_LEVEL_POSONLY in self.positional_encoding_:
			self.positional_encoding[1] = PositionalEncoding(hidden_size, only_positions=True)

		decoder_layer = TransformerDecoderLayer(
			hidden_size, self.num_heads, batch_first=True, dropout=self.dropout
		)
		self.decoder = TransformerDecoder(decoder_layer, num_layers=self.depth)

		decoder_layer = TransformerDecoderLayer(
			hidden_size, self.num_heads, batch_first=True, dropout=self.dropout
		)
		self.anomaly_decoder = TransformerDecoder(decoder_layer, num_layers=self.depth)

		# if self.Projections.OUTPUT in self.projections:
		#     self.output_projection = Linear(hidden_size, feature_size)
		# else:
		#     assert hidden_size == feature_size, (
		#         "A projection of the features back to the model's embedding "
		#         "dimension is needed if the specified hidden dimension is "
		#         "different."
		#     )

	def encode(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
		"""Encode the inputs to means and "variances" (`log(sigma**2)`).

		Arguments:
			inputs (Tensor): The input data.

		Returns:
			Tuple[Tensor, Tensor]: Mean and variances for each sample in the
				batch.
		"""
		if self.positional_encoding[0]:
			inputs = self.positional_encoding[0](inputs)

		features : Tensor = self.encoder(inputs)
		match self.sampling_level:
			case self.SamplingLevel.MEAN:
				features_1d = features.mean(dim=1)
			case self.SamplingLevel.ALL:
				features_1d = features.reshape(-1, features.size(-1))

		mean = self.encoder_mu(features_1d)
		log_varsq = self.encoder_sigma(features_1d)

		return mean, log_varsq

	def reparametrize(self, mean: Tensor, log_varsq: Tensor) -> Tensor:
		"""Reparametrisation trick, to backpropagate when using randomness.

		Arguments:
			mean (Tensor): Samples means.
			log_varsq (Tensor): Samples variances.

		Returns:
			Tensor: Sampled random vectors.
		"""
		std = (log_varsq * 0.5).exp()
		eps = torch.randn_like(std)

		return mean + std * eps

	def forward(self, inputs: Tensor) -> Tensor:
		source_inputs = inputs.detach()
		mean, log_varsq = self.encode(source_inputs)

		sampled = self.reparametrize(mean, log_varsq)
		# if self.hidden_projection:
		#     sampled = self.hidden_projection(sampled)

		if isinstance(self.parameters_loss, KLDivergence):
			self.loss = self.parameters_loss(set1=mean, set2=log_varsq)
		else:
			self.loss = self.parameters_loss(
				set1=sampled, set2=torch.rand_like(sampled)
			)

		if self.sampling_constraint is not None:
			self.loss = self.loss - self.sampling_constraint(batch=sampled)

		match self.sampling_level:
			case self.SamplingLevel.MEAN:
				sampled = sampled.unsqueeze(1).expand(-1, inputs.size(1), -1)
			case self.SamplingLevel.ALL:
				sampled = sampled.reshape(*inputs.size())

		target_inputs = inputs.detach()
		if self.positional_encoding[1]:
			target_inputs = self.positional_encoding[1](target_inputs)

		reconstructed = self.decoder(target_inputs, sampled)
		if self.secondary_task is self.SecondaryTask.RECONSTRUCTION:
			self.loss = self.loss + self.reconstruction_loss(reconstructed, inputs)

		outputs = self.anomaly_decoder(target_inputs, self.resampling_strategy(sampled))

		if self.outputs_constraint is not None:
			self.loss = self.loss - self.outputs_constraint(batch=outputs)

		# if self.output_projection:
		#     outputs = self.output_projection(outputs)

		if self.residual:
			return inputs + outputs

		return outputs

class FineGrainedTuning(Flag):
	"""Which fine-tuning strategy to apply to the LLM.

	Attributes:
		NONE: Does not apply any tuning.
		ALL: Fine-tune all parameters.
		LAYER_NORM: Fine-tune the LayerNorm layers.
		POSITIONAL: Fine-tune the positional encoding layer.

	NONE and ALL flags cannot be set with any other flag. Other flags can be
	combined ('|'). Using the `&` operator with NONE will cancel all other
	flags (return NONE); with ALL, will return ALL.
	"""

	NONE = auto()
	ALL = auto()
	LAYER_NORM = auto()
	POSITIONAL = auto()

	__EXCLUSIVE = [
		NONE | ALL | LAYER_NORM,
		NONE | ALL | POSITIONAL,
	]

	def __or__(self, other: "FineGrainedTuning") -> "FineGrainedTuning":
		combination = super().__or__(other)
		for excluded in self.__EXCLUSIVE:
			assert len(combination & FineGrainedTuning(excluded)) <= 1, (
				f"{self.__class__.__name__}: {self.name} and {other.name} are "
				"mutually exclusive flags. Choose only one."
			)

		return combination

	def __and__(self, other: "FineGrainedTuning") -> "FineGrainedTuning":
		if self is FineGrainedTuning.NONE or other is FineGrainedTuning.NONE:
			return FineGrainedTuning.NONE

		if self is FineGrainedTuning.ALL:
			return other

		if other is FineGrainedTuning.ALL:
			return self

		return super().__and__(other)

	@classmethod
	def from_bool(  # pylint: disable=missing-param-doc,missing-return-doc
		cls, value: bool
	) -> "FineGrainedTuning":
		"""Convert a boolean to a FineGrainedTuning flag."""
		return cls.ALL if value else cls.NONE

class BaseLLM(Module, metaclass=ABCMeta):
	"""Base class for LLMs."""

	@property
	@abstractmethod
	def EMBEDDING_SIZE(  # pylint: disable=no-self-argument,invalid-name
		self,
	) -> int:
		"""The embedding size of the LLM."""

	def __init__(self, cache_dir: str | None = None) -> None:
		super().__init__()

		with ensure_hf_cache(cache_dir):
			self.model = self._HF_CLASS.from_pretrained(
				self._PRETRAINED_PATH, cache_dir=cache_dir
			)
		# TODO: bypass position embedding layer?

	@abstractmethod
	def forward(self, embeddings: Tensor, *args: Any, **kwargs: Any) -> Tensor:
		"""Forward pass.

		Arguments:
			embeddings (Tensor): The inputs processed by an adapter.
			*args (Any): Positional arguments.
			**kwargs (Any): Keyword arguments.

		Returns:
			Tensor: The features extracted by the LLM.
		"""

	def tunable_parameters(
		self,
		fine_tune: FineGrainedTuning = FineGrainedTuning.NONE,  # pylint: disable=unused-argument
	) -> Iterable[Parameter]:
		"""List of the LLM parameters that can be tuned.

		Arguments:
			fine_tune (FineGrainedTuning): The fine-grained tuning options.
				Defaults to NONE (no fine tuned parameters).

		Returns:
			Iterable[Parameter]: The list of parameters. Default is empty.
		"""
		return []

	@property
	@abstractmethod
	def _PRETRAINED_PATH(self) -> str:  # pylint: disable=invalid-name
		"""HuggingFace pre-trained model path."""

	@property
	@abstractmethod
	def _HF_CLASS(  # pylint: disable=invalid-name
		self,
	) -> Type[PreTrainedModel]:
		"""HuggingFace model class."""

	def _lora_target_parameters(  # pylint: disable=missing-return-doc
		self,
	) -> List[str]:
		"""Names of parameters to be targeted with LoRA."""
		return []

class AdaptedLLMSubModule(Module, metaclass=ABCMeta):
	"""Base class for the adapters and heads."""

	def __init__(self) -> None:
		super().__init__()

		self.loss: Tensor
		self.register_buffer("loss", torch.tensor(0.0))

class BaseAdapter(AdaptedLLMSubModule, metaclass=ABCMeta):
	"""Base adapter class to adapt inputs to an LLM embedding space."""

	def __init__(self) -> None:
		super().__init__()

		self.embedding_size: int

	@abstractmethod
	def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
		"""Forward pass.

		Arguments:
			inputs (Tensor): The input data.
			*args (Any): Positional arguments.
			**kwargs (Any): Keyword arguments.

		Returns:
			Tensor: The adapted inputs.
		"""

	def setup_with_llm(self, llm: BaseLLM) -> None:
		"""Setup with the given llm.

		Arguments:
			llm (BaseLLM): The LLM from which to get its embedding size.
		"""
		self.embedding_size = llm.EMBEDDING_SIZE


class FeatureExtractor(Module, metaclass=ABCMeta):
	"""Base class for feature extractors."""

	def __init__(self, layer: int | None = None) -> None:
		"""Initialise the feature extractor.

		Arguments:
			layer int | None: If features can be extracted from multiple layers,
				a specific layer can be chosen to only return the features from
				the desired level. Defaults to None.
		"""
		super().__init__()

		self.layer = layer

		self.loss: Tensor
		self.register_buffer("loss", torch.tensor(0.0))

	@property
	@abstractmethod
	def embedding_sizes(self) -> List[Tuple[int, int]]:
		"""List of shapes of extracted features.

		Returns:
			List[Tuple[int, int]]: List of shapes (number of channels, feature
				size) of the extracted features. Number of channels can depend
				on the window size (e.g., one channel is one time step).
		"""

	@abstractmethod
	def forward(
		self, inputs: Tensor, *args: Any, **kwargs: Any
	) -> List[Tensor]:
		"""Forward pass.

		Arguments:
			inputs (Tensor): The inputs.
			*args (Any): Positional arguments.
			**kwargs (Any): Keyword arguments.

		Returns:
			List[Tensor]: Features from at least one depth.
		"""

	def tunable_parameters(self) -> Iterable[Parameter]:
		"""List of the feature extractor parameters that can be tuned.

		Returns:
			Iterable[Parameter]: The list of parameters. Default is empty.
		"""
		return []

class AdaptedLLM(FeatureExtractor):
	"""LLM with an adapter and a head."""

	def __init__(
		self,
		win_size: int,
		adapter: "BaseAdapter",
		llm: "BaseLLM",
		head: Optional[Any] = None,
		contextualise: bool = True,
		fine_tune: FineGrainedTuning = FineGrainedTuning.NONE,
		**kwargs: Any,
	) -> None:
		super().__init__(**kwargs)

		self.adapter = adapter
		self.model = llm
		self.head = head
		self.contextualise = contextualise
		self.fine_tune = FineGrainedTuning.from_bool(contextualise) & fine_tune

		self.window_size: int = win_size

	@property
	def embedding_sizes(self) -> List[Tuple[int, int]]:
		"""List of shapes of extracted features.

		Shapes of features from the adapter, the LLM and the head. If `layer`
		has been given, return a list with one element: the shape at that layer.

		Returns:
			List[Tuple[int, int]]: List of shapes (number of channels, feature
				size) of the extracted features. Number of channels can depend
				on the window size (e.g., one channel is one time step).
		"""
		sizes = (
			self.adapter.embedding_sizes
			+ (
				[(self.window_size, self.model.EMBEDDING_SIZE)]
				if self.contextualise
				else []
			)
			+ (self.head.embedding_sizes if self.head is not None else [])
		)

		return sizes if self.layer is None else [sizes[self.layer]]

	# def setup_with_datamodule(self, datamodule: BaseDataModule) -> None:
	#     self.window_size = datamodule.sample()[0].shape[1]

	def forward(
		self, inputs: Tensor, *args: Any, **kwargs: Any
	) -> List[Tensor]:
		"""Forward pass.

		Arguments:
			inputs (Tensor): The inputs to the model.
			*args (Any): Positional arguments.
			**kwargs (Any): Keyword arguments.

		Returns:
			List[Tensor]: The features after the adapter, after the LLM, and
				the ones from the head if any. If `layer` has been given at
				initialisation, returns a list with a unique element, the Tensor
				of features at `layer` index (omitting the predictions of the
				head, i.e., `layer=-1` will return the last available features
				before the head predictions).
		"""
		adapted = self.adapter(inputs)
		contextualised = self.model(adapted) if self.contextualise else adapted

		self.loss = self.adapter.loss

		projected = [adapted] + ([contextualised] if self.contextualise else [])
		predictions = []
		if self.head is not None:
			outputs = self.head(contextualised)
			self.loss += self.head.loss

			projected += outputs[:-1]
			predictions += outputs[-1:]

		return (
			projected + predictions
			if self.layer is None
			else [projected[self.layer]]
		)

	def tunable_parameters(self) -> Iterable[Parameter]:
		"""List of the feature extractor parameters that can be tuned.

		Always include the adapter parameters, the LLM parameters if a fine
		tuning was requested, and the head parameters if any provided.

		Returns:
			Iterable[Parameter]: The list of parameters.
		"""
		return itertools.chain(
			self.adapter.parameters(),
			self.model.tunable_parameters(self.fine_tune),
			(self.head.parameters() if self.head is not None else []),
		)

class Projector(FeatureExtractor):
	"""Simple projector, without LLM."""

	def __init__(self, adapter: BaseAdapter, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, layer=0, **kwargs)

		self.adapter = adapter

	@property
	def embedding_sizes(self) -> List[Tuple[int, int]]:
		return self.adapter.embedding_sizes

	def forward(
		self, inputs: Tensor, *args: Any, **kwargs: Any
	) -> List[Tensor]:
		"""Forward pass.

		Arguments:
			inputs (Tensor): The inputs.
			*args (Any): Positional arguments.
			**kwargs (Any): Keyword arguments.

		Returns:
			List[Tensor]: Features from at least one depth.
		"""
		adapted = self.adapter(inputs)
		
		self.loss = self.adapter.loss

		return [adapted]

	def tunable_parameters(self) -> Iterable[Parameter]:
		"""List of the feature extractor parameters that can be tuned.

		Returns:
			Iterable[Parameter]: The list of parameters. Default is empty.
		"""
		return list(self.adapter.parameters())

class LoRA(BaseLLM):
	"""LoRA tuning on top of another LLM implementation."""

	@property
	def EMBEDDING_SIZE(  # pylint: disable=invalid-name,missing-return-doc
		self,
	) -> int:
		return self.model_.EMBEDDING_SIZE

	@property
	def _PRETRAINED_PATH(self) -> str:  # pylint: disable=missing-return-doc
		return self.model_._PRETRAINED_PATH  # pylint: disable=protected-access

	@property
	def _HF_CLASS(  # pylint: disable=missing-return-doc
		self,
	) -> Type[PreTrainedModel]:
		return self.model_._HF_CLASS  # pylint: disable=protected-access

	def __init__(
		self,
		model_class: Type[BaseLLM],
		rank: int = 10,
		cache_dir: str | None = None,
	) -> None:
		# Avoiding BaseLLM.__init__(), bypassing model load.
		# pylint: disable=non-parent-init-called,super-init-not-called
		Module.__init__(self)
		# Base.__init__(self)

		self.model_ = model_class(cache_dir=cache_dir)

		config_ = LoraConfig(
			r=rank,
			target_modules=self.model_._lora_target_parameters(),
			task_type=TaskType.FEATURE_EXTRACTION,
			lora_alpha=32,
			lora_dropout=0.1,
		)

		self.model = get_peft_model(self.model_.model, config_)

	def forward(self, embeddings: Tensor, *args: Any, **kwargs: Any) -> Tensor:
		"""Forward pass.

		Arguments:
			embeddings (Tensor): The inputs processed by an adapter.
			*args (Any): Positional arguments.
			**kwargs (Any): Keyword arguments.

		Returns:
			Tensor: The outputs of the llm, that is one new "attended" embedding
				per each inputs.
		"""
		attended_embeddings: BaseModelOutputWithPastAndCrossAttentions = (
			self.model(
				inputs_embeds=embeddings,
				output_hidden_states=False,
				use_cache=False,
			)
		)

		return attended_embeddings.last_hidden_state

	def tunable_parameters(  # pylint: disable=unused-argument
		self, fine_tune: FineGrainedTuning = FineGrainedTuning.NONE
	) -> Iterable[Parameter]:
		"""List of the LoRA parameters

		Arguments:
			fine_tune (FineGrainedTuning): The fine-grained tuning options.
				Defaults to NONE (no fine tuned parameters).

		Returns:
			Iterable[Parameter]: List of LoRA parameters and the chosen LLM
				parameters to be tuned
		"""
		return [
			v for k, v in self.model.state_dict().items() if "lora" in k
		] + list(self.model_.tunable_parameters(fine_tune=fine_tune))


class GPT2(BaseLLM):
	"""GPT2 support as a BaseLLM."""

	@property
	def EMBEDDING_SIZE(  # pylint: disable=no-self-argument,invalid-name,missing-return-doc
		self,
	) -> int:
		return 768

	@property
	def _PRETRAINED_PATH(self) -> str:  # pylint: disable=missing-return-doc
		return "gpt2"

	@property
	def _HF_CLASS(  # pylint: disable=missing-return-doc
		self,
	) -> Type[PreTrainedModel]:
		return GPT2Model

	def forward(self, embeddings: Tensor, *args: Any, **kwargs: Any) -> Tensor:
		"""Forward pass.

		Arguments:
			embeddings (Tensor): The inputs processed by an adapter.
			*args (Any): Positional arguments.
			**kwargs (Any): Keyword arguments.

		Returns:
			Tensor: The outputs of the llm, that is one new "attended" embedding
				per each inputs.
		"""
		attended_embeddings: BaseModelOutputWithPastAndCrossAttentions = (
			self.model(
				inputs_embeds=embeddings,
				output_hidden_states=False,
				use_cache=False,
			)
		)

		return attended_embeddings.last_hidden_state

	def tunable_parameters(
		self, fine_tune: FineGrainedTuning = FineGrainedTuning.NONE
	) -> Iterable[Parameter]:
		"""List of the GPT2 parameters that will be tuned.

		Arguments:
			fine_tune (FineGrainedTuning): The fine-grained tuning options.
				Defaults to NONE (no fine tuned parameters).

		Returns:
			Iterable[Parameter]: Either an empty list (NONE), all parameters
			(ALL), or `wpe` and `ln*` parameters (POSITIONAL and LAYER_NORM).
		"""
		if fine_tune is FineGrainedTuning.NONE:
			return []

		if fine_tune is FineGrainedTuning.ALL:
			return self.parameters()

		parameters = []
		for name, param in self.named_parameters():
			if "wpe" in name and FineGrainedTuning.POSITIONAL in fine_tune:
				parameters.append(param)
			elif "ln" in name and FineGrainedTuning.LAYER_NORM in fine_tune:
				parameters.append(param)

		return parameters

	def _lora_target_parameters(  # pylint: disable=missing-return-doc
		self,
	) -> List[str]:
		"""Names of parameters to be targeted with LoRA."""
		return ["c_attn", "c_proj"]
		# return ["ln", "wpe"]



class Normalise(Enum):
	BATCH_NORM = auto()
	LAYER_NORM = auto()
	NONE = auto()


class LinearAdapter(BaseAdapter):
	def __init__(
		self,
		num_signal: int,
		window_size: int,
		emb_size: int,
		isolated_projection: bool = True, **kwargs: Any
	) -> None:
		super().__init__(**kwargs)

		self.isolated_projection = isolated_projection
		self.window_size = window_size
		self.embedding_size = emb_size

		factor = 1 if self.isolated_projection else window_size

		input_size = factor * num_signal
		output_size = factor * self.embedding_size

		self.dense = Linear(in_features=input_size, out_features=output_size)

	@property
	def embedding_sizes(self) -> List[Tuple[int, int]]:
		"""List of shapes of extracted features.

		Shape of features after the MLP.

		Returns:
			List[Tuple[int, int]]: List of shapes (window_size, feature size) of
				the extracted features.
		"""
		return [(self.window_size, self.embedding_size)]

	def forward(self, inputs: Tensor, *args: Any, **kwargs: Any) -> Tensor:
		"""Forward pass.

		Outputs embeddings with the context size equal to the window size.

		Arguments:
			inputs (Tensor): The input data.
			*args (Any): Positional arguments.
			**kwargs (Any): Keyword arguments.

		Returns:
			Tensor: The projected inputs.
		"""
		batch_size, window_size, _ = inputs.shape
		reshape = (
			(batch_size * window_size, -1)
			if self.isolated_projection
			else (batch_size, -1)
		)

		outputs: Tensor = self.dense(inputs.reshape(*reshape))

		return outputs.reshape(batch_size, window_size, self.embedding_size)

class ASTER(Module):
	class _ReverseGradient(Enum):
		FALSE = auto()
		TRUE = auto()
		WARM_START = auto()

	def __init__(
		self,
		window_size: int,
		num_features: int,
		signal_num: int,
		*args: Any,
		stop_grad: bool = True,
		**kwargs: Any,
	) -> None:
		super().__init__(*args, **kwargs)
		
		# super().__init__(**kwargs)
		self.signal_num = signal_num  # number of signals in dataset

		self.perturbator = EDTransformerVAE(
			parameters_loss=KLDivergence(),
			reconstruction_loss=MSELoss(),
			resampling_strategy=NoReSampling(),
			positional_encoding=EDTransformerVAE.PositionalEncoding_.SOURCE_LEVEL | EDTransformerVAE.PositionalEncoding_.TARGET_LEVEL_POSONLY,
			depth=2,
			num_heads=4,
			feature_size=num_features,
		)
		self.classifier = Transformer(
			emb_size=num_features,
			depth=2,
			pooling=PoolingType.CLS,
			positional_encoding=True,
			num_heads=4,
		)
		
		self.llm_lora = True # TODO: make parametric 
		self.use_llm = True # TODO: make parametric 
		
		if self.use_llm:
			if self.llm_lora:
				llm = LoRA(
					model_class=GPT2,
					cache_dir="./pretrained",
				)
			else:
				llm = GPT2(
					cache_dir="./pretrained"
				)
			
			self.extractor = AdaptedLLM(
				win_size=window_size,
				adapter=LinearAdapter(emb_size=num_features, num_signal=self.signal_num, window_size=window_size),
				layer=1,
				llm=llm,
				# fine_tune=FineGrainedTuning.LAYER_NORM | FineGrainedTuning.POSITIONAL,
			)
		else:
			self.extractor = Projector(
			    adapter=LinearAdapter(num_signal=signal_num, window_size=window_size, emb_size=num_features)
			)

		self.stop_gradient = StopGradient() if stop_grad else DummyModule()

		self.reverse_gradient = ReverseGradient()

	def _training_procedure(self, inputs: Tensor) -> Tensor:
		"""Common procedure for training and validation.

		Arguments:
			inputs (Tensor): The normal inputs to the model.

		Returns:
			Tensor: The output of the classifier.
		"""
		pseudo_anomalies: Tensor = self.perturbator(self.stop_gradient(inputs))

		inputs_ = torch.concatenate(
			[inputs.reshape(*pseudo_anomalies.shape), self.reverse_gradient(pseudo_anomalies)]
		)

		predictions = self.classifier(inputs_)

		self.loss = self.perturbator.loss + self.classifier.loss

		return predictions

	def _apply_extraction(self, inputs: Tensor) -> Tensor:
		"""Extract the data and adapt the features, if necessary.

		Arguments:
			inputs (Tensor): The inputs to the model.

		Returns:
			Tensor: Extracted input if an extractor was provided, adapted inputs
				if an adapter was provided, the same tensor otherwise.
		"""
		if self.extractor:
			inputs = self.extractor(inputs)[-1]

		return inputs
	
	@staticmethod
	def _adapt_targets(targets: Tensor) -> Tensor:
		"""Change the original targets for ASTER's self-supervision.

		ASTER will create `len(targets)` pseudo-anomalies from `len(targets)`
		normal samples, hence the targets need to be changed to zeros and ones:
		[len(targets) * normal (0), len(targets) * abnormal (1)].

		Arguments:
			targets (Tensor): The targets to be adapted.

		Returns:
			Tensor: The new targets, twice the length of the originals.
		"""
		return torch.concatenate(
			[
				torch.zeros(
					len(targets), device=targets.device, dtype=targets.dtype
				),
				torch.ones(
					len(targets), device=targets.device, dtype=targets.dtype
				),
			]
		)

	def training_step(
		self,
		batch: Tuple[Tensor, Tensor],
	) -> Tensor:
		"""ASTER's training step.

		Extract and adapt the data if needed, create the pseudo-anomalies with
		the perturbator, try to classify the normal data and the
		pseudo-anomalies and compute the total loss.

		Arguments:
			batch (Tuple[Tensor, Tensor]): The current batch.
			batch_idx (int): The batch index.
			dataloader_idx (int): The dataloader index. Defaults to 0.

		Returns:
			Tensor: The total loss for the current batch.
		"""
		# inputs = batch
		inputs = self._apply_extraction(batch)
		targets_ = self._adapt_targets(batch)

		predictions = self._training_procedure(inputs)

		return predictions, targets_, self.loss 
	

class ActionModule(Module):
	pass
	# _linked_model: "BaseModule"

	# def setup_with_model(self, model: "BaseModule") -> None:
	#     """Setup the module with the model."""
	#     self._linked_model = model

	# def __setattr__(self, name: str, value: Tensor | Module) -> None:
	#     """Ensure linked modules are not added recursively to the parameters.

	#     A link to the model is needed in the submodules, but should not be
	#     added in the submodule's parameter list.
	#     """
	#     if name == "_linked_model":
	#         return object.__setattr__(self, name, value)

	#     return super().__setattr__(name, value)
	
class StopGradient(ActionModule):
	class StopGradientFunction(Function):
		@staticmethod
		def forward(ctx: Any, input: Tensor) -> Tensor:
			return input

		@staticmethod
		def backward(ctx: Any, grad_output: Tensor) -> Tensor:
			return torch.zeros_like(grad_output)

	def forward(self, inputs: Tensor) -> Tensor:
		return self.StopGradientFunction.apply(inputs)

class ReverseGradient(ActionModule):
	LOG_KEY = "reverse_gradient_coeff"

	def __init__(
		self,
		*args,
		max_epochs: int = -1,
		steps_per_epoch: int = -1,
		acc_grad: int = 1,
		warm_start: bool = False,
		**kwargs,
	):
		super().__init__(*args, **kwargs)

		self._acc_grad = acc_grad
		self._steps_per_epoch = math.ceil(steps_per_epoch / self._acc_grad)
		self._max_epochs = max_epochs
		self._warm_start = warm_start
		
		self._current_epoch = 0
		self._current_batch = -1
		self._current_update = -1

		self.coeff = 1.0

	class ReverseGradientFunction(Function):
		@staticmethod
		def forward(ctx: Any, input: Tensor, coeff: float) -> Tensor:
			ctx.coeff = coeff

			return input

		@staticmethod
		def backward(ctx: Any, grad_output: Tensor) -> Tensor:
			return -ctx.coeff * grad_output, None
		
	def _calculate_coeff(self) -> float:
		# https://github.com/BIT-DA/MsRA/blob/main/MsRA.py
		if self._max_epochs <= 0 or self._steps_per_epoch <= 0 or not self._warm_start:
			return self.coeff

		self._current_batch += 1
		if self._current_batch % self._acc_grad == 0:
			self._current_update += 1
		
		if self._current_batch >= self._steps_per_epoch:
			self._current_epoch += 1

		coeff_ = torch.tensor(
			self._current_update / (self._max_epochs * self._steps_per_epoch)
		)
		self.coeff = (2.0 / (1.0 + torch.exp(-10.0 * coeff_)) - 1.0).item()

		# self._linked_model.log(
		#     self.LOG_KEY, self.coeff, sync_dist=True, add_dataloader_idx=False
		# )

		return self.coeff
	
	def reset(self) -> None:
		self._current_epoch = 0
		self._current_batch = -1
		self._current_update = -1

	def forward(self, inputs: Tensor) -> Tensor:
		return self.ReverseGradientFunction.apply(
			inputs, self._calculate_coeff()
		)

class DummyModule(ActionModule):
	def forward(self, inputs: Tensor) -> Tensor:
		return inputs

class BCEWithLogitsLoss(BaseLoss):
	"""Binary Cross-Entropy from logits."""

	LOSS = "bce"

	def __init__(self, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)

		self._loss = TorchBCEWithLogitsLoss()

	def __call__(self, set1: Tensor, set2: Tensor, **kwargs: Any) -> Tensor:
		loss = self._loss(set1, set2)

		# self.log(loss)

		return self.coefficient * loss
