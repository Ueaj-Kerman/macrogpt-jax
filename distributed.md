## Abstractions for Distributed Training

- [x] Distributed Data Loader
    - [x] Check if the current host has the first device along the batch axis (i.e. pos 0 along the other axes)
        - `distutil.this_host_has_first(mesh, 'data')`
    - [ ] No more bin packing now that we have sparse attention
    - [ ] Checkpointing support
        - each host counts the rows for one shard in parallel
        - use this to approximately distribute starting points of each data loader
        - checkpoint last returned sequence offset so we can restart there
- [ ] Distributed Model Initialization
    - [x] Initialize model on first batch rank
        - `distutil.slice(mesh)[:1, :1, ...]` 
        - slices a mesh along axes to conveniently select first batch & sequence rank of mesh
    - Reshard into master weights
        - `distutil.block_allocations(model, mesh, group_einsum_tensors=True)` (probably tuples of (tensor_id,
          split_id))
            - mesh is sharded along ('tensors', 'blocks') axis
            - obtain einsum tensors, map into (...batch_dims, reducing_dims, non_reducing_dims) format
            - given size of 'blocks' axis, find most square block tile size of tensor's (reducing_dims,
              non_reducing_dims)
            - record maximum block tile size (w x h), this is block alloc size
            - find number of blocks needed for the other parameters
            - allocate tensor array ('tensors', 'blocks') large enough to assign one tensor index to each einsum tensor
              and enough extra tensor indices to assign blocks to the other parameters
            - basically for each tensor, give it a tuple of ([(tensor_id, block_id), ...], (block_alloc_w,
              block_alloc_h)) where the list of tensor_id and block_id also contains the assignments in case the tensor
              has a batch axis (i.e. fused matmul or vmapped layers)
            - if needed, carry over any metadata from einsum tensor in the returned value for blockify/debloky
        - `distutil.blockify(allocations, state)`, `distutil.deblokify(state)`
          - split state, pad, stack
          - opposite of that
        - `distutil.shard(blocks, block_allocs, mesh)`
    - Sharded optimizer
        - more or less handled by `distutil.blockify`
        - shard_map gradients, blockify, sum, shard, pass to optimizer


1. how good are sharding constraints? i.e. if I .sum across my gradients and then place a sharding constraint equivalent
   to that of a psum scatter, will it actually make it a psum scatter?
2. how good is the compute-comms overlap in the compiler, if I shove my entire model state into one big tensor and shard
   it, will it really overlap the partial writes in the backwards pass?
3. how to I avoid large compile times and still gain the compute-comms overlap, to my knowledge if I use a scan it won't
   overlap
4. In single host environments using a scan will prevent the optimizer pass from being fused into the backwards pass