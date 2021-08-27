# Development Guide


This is a guide on the development of a new algorithm.

To develop a new algorithm, follow the steps below.

1. Add your parameter script to ``common/frame/parameter_frame/``, subclassing ``Parameter``.
2. Add your output model script to ``common/frame/model_frame/``, subclassing ``Model``.
3. Wrap your list of federated messages to a script
   and add it to ``common/frame/message_frame/``, subclassing ``Message``.
4. Add your algorithm script to ``fl/``, subclassing ``Algorithm``, and your privacy-preserving techniques
   to ``privacy/`` if have. Please note that the following abstract
   methods must be overridden
   - ``Algorithm.train()``
   - ``Algorithm.predict()``
   - ``Algorithm.model_to_instance()``
   - ``Algorithm.instance_to_model()``
5. Add your algorithm and parameter name to ``common/factory/algorithm_factory.py``, ``common/factory/m2i_factory.py``
   and ``common/factory/parameter_factory.py``, respectively.
6. Update the factories if new privacy-preserving techniques are presented in your algorithm, such as encryptor (``common/factory/encryptor_factory``) and dp_encoder (``common/factory/dp_factory``).
7. Add an example of your algorithm configuration to ``example/conf/``.
8. Add your algorithm description to the Use Guide section in``README.md``.
9. (Optional) Add your algorithm description to ``doc/algorithm`` for readers' better understanding.

To test your algorithm, refer to ``doc/guide/deploy_guide.md``.
