from __future__ import annotations

import bisect
import heapq
import pickle
import random
from collections import defaultdict
from typing import Any, Callable, Generator, Iterator, List, Tuple

import numpy as np

import 赛博蚊香  # 保佑我别出bug # noqa: F401


class LinearTrie:
    """线性表示的Trie树，用于以时间换空间地完成Trie树的功能，同时支持通配符。

    Methods:
        feistel_rsg(start, end, seed=None):
            基于Feistel算法的RSG（随机序列生成器）。
        numpy_rsg(start, end, seed=None):
            基于numpy的RSG。
        shuffled(seq, stream=True, seed=None):
            对序列进行随机打乱。当stream为True时，采用Feistel算法流式生成。
            如果不需要流式生成，请不要使用stream=True。Feistel算法在非流式情况下效率远低于NumPy。
            是否流式生成不会影响其返回格式。无论是否流式生成，shuffled始终返回一个生成器。
        get(target, shuffle=False, stream=True)
            返回前缀搜索的结果。
            当stream为True时，将采用流式生成的算法，此算法对于全量生成来说非常低效。 因此，如果需要全量搜索，请使用stream=False。
            是否流式生成不会影响其返回格式。无论是否流式生成，get始终返回一个str生成器。"""

    UMAX = "\U0010ffff"  # 最大UNICODE码点。虽然说用于边界的话它本身不会被匹配，但是也无所谓了……
    UMIN = "\u0000"  # 最小UNICODE码点
    _DEFAULT_WILDCARD = "\u0000"  # 默认通配符，会覆盖初始化时给定的\u0000通配（如有）
    _DEFAULT_WILDCARE_RANGE = [(UMIN, UMAX)]  # 默认通配符的匹配范围
    __slots__ = ("wildcard_map", "sorted_seq", "range_index")

    def __init__(self, sorted_seq: tuple[str, ...], wildcard_map: dict[str, list[tuple[str, str]]] | None = None):
        """创建一个线性Trie树。

        Args:
            sorted_seq (tuple[str, ...]):
                有序的字符串序列（元组格式）。无序不会导致致命错误，但是可能返回错误结果。
                出于性能考虑，并未内置排序，因此请务必保证输入有序。
            wildcard_map (dict[str, list[tuple[str, str]]] | None, optional):
                通配符映射表。格式为{w:[(s1,e1),(s2,e2),...],...}。
                其中w为通配符，s e为匹配范围。注意含s不含e，因此("a","a")什么都匹配不到，("a","b")只能匹配到"a"。
                w s e都应当为单字符。如果不为单字符，不会导致致命错误，但是可能返回错误结果。
                通配符默认为u+0000，匹配整个UNICODE范围。
        """
        self.wildcard_map = wildcard_map or {}
        self.wildcard_map.update({LinearTrie._DEFAULT_WILDCARD: LinearTrie._DEFAULT_WILDCARE_RANGE})
        self.sorted_seq = sorted_seq

    def __getitem__(self, target: str) -> list[str]:
        return list(self.get(target))

    @staticmethod
    def feistel_rsg(start: int, end: int, seed: Any = None) -> Generator[int, None, None]:
        """基于Feistel算法的RSG（随机序列生成器）。

        Args:
            start (int): 起始值（含）。
            end (int): 终止值（不含）。
            seed (Any, optional): 种子。默认为None。

        Yields:
            Generator[int, None, None]: 随机序列。
        """
        # 我不知道这个算法具体是咋实现的，也不知道为啥能生成全排列，我只是上网找了个教程现学现用的（爬）
        # 但绝对不是照抄或者CV的
        span = end - start
        rng = random.Random(seed) if seed is not None else random

        bitlen = (span - 1).bit_length()
        hi = (bitlen + 1) // 2
        lo = bitlen - hi
        mask_hi = (1 << hi) - 1
        mask_lo = (1 << lo) - 1

        rounds = (rng.randint(0, mask_hi), rng.randint(0, mask_lo), rng.randint(0, mask_hi))

        def _feistel(x):
            ll = (x >> lo) & mask_hi
            rr = x & mask_lo
            ll, rr = rr, ll ^ ((rr + rounds[0]) & mask_hi)
            ll, rr = rr, ll ^ ((rr ^ rounds[1]) & mask_lo)
            ll, rr = rr, ll ^ ((rr + rounds[2]) & mask_hi)
            return (ll & mask_hi) << lo | (rr & mask_lo)

        cache = {}
        for i in range(span):
            x = i
            while True:
                if x in cache:
                    result = cache[x]
                    del cache[x]
                    break

                x = _feistel(x)
                if x < span:
                    result = x
                    if i != x:
                        cache[i] = x
                    break
            yield start + result

    @staticmethod
    def numpy_rsg(start: int, end: int, seed: Any = None) -> Generator[int, None, None]:
        """NumPy RSG的套壳。

        Args:
            start (int): 起始值（含）。
            end (int): 终止值（不含）。
            seed (Any, optional): 种子。默认为None。

        Yields:
            Generator[int, None, None]: 随机序列。
        """
        arr = np.arange(start, end)
        rng = np.random.default_rng(seed)
        rng.shuffle(arr)
        yield from arr

    @staticmethod
    def shuffled(lst: Any, stream: bool = True, seed: Any = None) -> Generator[Any, None, None]:
        """采用Feistel或NumPy RSG之一对列表进行打乱。

        Args:
            lst (Any): 待打乱的列表。
            stream (bool, optional): 是否流式生成。默认为True。
            seed (Any, optional): 种子。默认为None。

        Yields:
            Generator[Any, None, None]: 打乱后的列表。

        注意: 此方法不是就地打乱！"""
        gen = LinearTrie.feistel_rsg(0, len(lst), seed) if stream else LinearTrie.numpy_rsg(0, len(lst), seed)
        for i in gen:
            yield lst[i]

    def get(self, target: str, shuffle: bool = False, stream: bool = True) -> Generator[str, None, None]:
        """返回前缀搜索的结果。

        Args:
            target (str): 目标字符串。
            shuffle (bool, optional): 是否打乱结果。默认为False。
            stream (bool, optional): 是否流式生成。默认为True。

        Yields:
            Generator[str, None, None]: 搜索结果。
        """
        if not target:
            yield from self.sorted_seq
            return
        levels = []
        tmp = ""
        w84n0 = False  # wait for not \u0000 太长了所以缩写一下很合理吧（不是
        for char in target:
            tmp += char
            if char == LinearTrie._DEFAULT_WILDCARD:
                w84n0 = True
                continue
            elif w84n0:  # 等到了第一个非零值
                levels.append((tmp[:-1], LinearTrie._DEFAULT_WILDCARD))
                w84n0 = False
                tmp = char
                continue
            if char in self.wildcard_map:
                levels.append((tmp, char))
                tmp = ""
        if tmp:  # 如果有剩下的无通配段
            levels.append((tmp, char if char == LinearTrie._DEFAULT_WILDCARD else None))

        spans = self._recur_gen("", levels, (0, len(self.sorted_seq)), shuffle=shuffle, stream=stream)
        for span_s, span_e in spans:
            segment = self.sorted_seq[span_s:span_e]  # 理论上流式模式下不用把这个切片切出来的，但我懒得优化了……
            if shuffle:
                segment = LinearTrie.shuffled(segment, stream=stream)
            yield from segment

    def _recur_gen(
        self, fixed: str, levels: List[Tuple[str, str | None]], span: tuple[int, int], shuffle: bool = False, stream: bool = True
    ) -> Generator[tuple[int, int], None, None]:
        # 简述一下设计思路，就是按照通配符切开保证通配符在每段的末尾，然后逐层二分匹配
        # 根据前(段长度)个字符分组之后，每个子分组内部必定有序，然后递归就行
        if span[0] >= span[1]:
            return
        if not levels:
            yield span
            return

        prefix, wildcard = levels[0]
        if wildcard is None:
            left = fixed + prefix
            right = fixed + prefix + LinearTrie.UMAX
            left_idx = bisect.bisect_left(self.sorted_seq, left, span[0], span[1])
            right_idx = bisect.bisect_right(self.sorted_seq, right, span[0], span[1])
            if left_idx == right_idx:
                return
            yield (left_idx, right_idx)
        else:
            multiple = wildcard == LinearTrie._DEFAULT_WILDCARD  # 如果有多个全量通配符则把这几个一起处理了
            replacement = self.wildcard_map[wildcard]
            fixedlen = len(fixed) + len(prefix)
            if shuffle:
                replacement = LinearTrie.shuffled(replacement, stream=stream)
            for start, end in replacement:
                left_bound = fixed + (prefix.replace(wildcard, start) if multiple else prefix[:-1] + start)
                right_bound = fixed + (prefix.replace(wildcard, end) if multiple else prefix[:-1] + end)
                left_idx = bisect.bisect_left(self.sorted_seq, left_bound, span[0], span[1])
                right_idx = bisect.bisect_right(self.sorted_seq, right_bound, span[0], span[1])
                if left_idx == right_idx:
                    continue
                if not levels[1:]:
                    yield (left_idx, right_idx)
                    continue

                gstart = left_idx
                gend = left_idx
                sstart = self.sorted_seq[left_idx][:fixedlen]
                generators = []

                if (span[1] - span[0]) > 4096 * fixedlen:  # 4096是试出来最快的，鬼才知道为啥，兴许换台设备就不是了呢……
                    while True:
                        gend = bisect.bisect_left(self.sorted_seq, sstart + LinearTrie.UMAX, gstart)
                        if gend >= right_idx:
                            break
                        seg_gen = self._recur_gen(sstart, levels[1:], (gstart, gend), shuffle)
                        if shuffle:
                            generators.append(seg_gen)
                        else:
                            yield from seg_gen
                        gstart = gend
                        sstart = self.sorted_seq[gstart][:fixedlen]
                    seg_gen = self._recur_gen(sstart, levels[1:], (gstart, right_idx), shuffle)
                    if shuffle:
                        generators.append(seg_gen)
                        for seg_gen in LinearTrie.shuffled(generators, stream=stream):
                            yield from seg_gen
                    else:
                        yield from seg_gen
                else:
                    for index, item in enumerate(self.sorted_seq[left_idx:right_idx]):
                        if not item.startswith(sstart):
                            gend = index + left_idx
                            seg_gen = self._recur_gen(sstart, levels[1:], (gstart, gend), shuffle)
                            if shuffle:
                                generators.append(seg_gen)
                            else:
                                yield from seg_gen
                            gstart = index + left_idx
                            sstart = item[:fixedlen]
                    seg_gen = self._recur_gen(sstart, levels[1:], (gstart, right_idx), shuffle)
                    if shuffle:
                        generators.append(seg_gen)
                        for seg_gen in LinearTrie.shuffled(generators, stream=stream):
                            yield from seg_gen
                    else:
                        yield from seg_gen


class WordList:
    """我懒的写了反正能看懂咋用就行

    >>> wl = WordList("词.csv","词.pkl")
    >>> len(wl["海"])
    123456
    >>> gen = wl.get("海\\u0000汤", shuffle=True, stream=True)
    >>> next(gen)
    "海带汤"
    >>> next(gen)
    "海龟汤"
    >>> wl["\\u0000海"]
    ["大海", "大海龟", "懒得举例了", ...]
    """

    BIG = 50_0000
    LARGE = 200_0000

    def __init__(self, raw: str | tuple[str, ...], src: str | None = None):
        """创建词库

        Args:
            raw (str | tuple[str, ...]): 原始文件路径或词表，如果是词表则要求有序元组
            src (str | None, optional): 缓存文件路径（可选），如果此路径内容有效那么raw无效
        """
        words = None
        if src is not None:
            try:
                with open(src, "rb") as f:
                    words: tuple[str, ...] | None = pickle.load(f)
            except FileNotFoundError:
                pass
        if words is None:
            if isinstance(raw, str):
                with open(raw, encoding="utf-8") as f:
                    content = f.read().strip()
                    if "\n" in content:
                        words = tuple(sorted(content.split("\n")))
                    else:
                        words = tuple(sorted(content))
                if src is not None:
                    with open(src, "wb") as f:
                        pickle.dump(words, f)
            else:
                words = tuple(sorted(raw))
        size = len(words)
        self.size = (size > WordList.LARGE) + (size > WordList.BIG)
        if self.size == 0:
            self.words: dict[int, tuple[str, ...]] = {0: words}
        else:
            w = defaultdict(list)
            for word in words:
                w[len(word)].append(word)
            self.words = {k: tuple(v) for k, v in w.items()}
        self._wildcard_map: dict[str, list[tuple[str, str]]] = {"\u0000": [("\u0000", "\U0010ffff")]}

    def __igetall(self) -> Generator[str, None, None]:
        return heapq.merge(*self.words.values())

    def __fromsize(self, size: int) -> tuple[str, ...]:
        if self.size == 0:
            return tuple(word for word in self.words[0] if len(word) == size)
        return self.words.get(size, ())

    def __len__(self) -> int:
        return sum(len(w) for w in self.words.values())

    def __iter__(self) -> Iterator[str]:
        return iter(self.__igetall())

    def __getitem__(self, index: int | str | slice | tuple) -> list[str]:
        lmin = 0
        lmax = max(self.words.keys()) + 1
        prefix = ""
        shuffle = False
        match index:  # 屎山警告
            case int(length):
                lmin = length
                lmax = length + 1
            case str(word):
                prefix = word
                lmin = len(word)
            case slice(start=start, stop=stop, step=shuffle):
                lmin = start if start is not None else lmin
                lmax = stop if stop is not None else lmax
                shuffle = shuffle is not None
            case (str(word), bool(shuffle)):
                prefix = word
                shuffle = shuffle
            case (str(word), int(length)):
                prefix = word
                lmin = length
                lmax = length + 1
            case (int(length), bool(shuffle)):
                lmin = length
                lmax = length + 1
                shuffle = shuffle
            case (str(word), slice(start=start, stop=stop, step=shuffle)):
                prefix = word
                lmin = start if start is not None else len(word)
                lmax = stop if stop is not None else lmax
                shuffle = shuffle is not None
            case _:
                raise TypeError(f"意外的索引: {repr(index)}")
        return list(self.get(prefix, lmin=lmin, lmax=lmax, shuffle=shuffle, stream=False))

    def __contains__(self, item: str):
        words = self.__fromsize(len(item))
        index = bisect.bisect_left(words, item)
        return index < len(words) and words[index] == item

    def get(
        self,
        target: str = "",
        quantity: int | None = None,
        lmin: int | None = None,
        lmax: int | None = None,
        shuffle: bool = False,
        stream: bool = True,
    ) -> Generator[str, None, None]:
        """参数是干啥的看参数名就行"""
        lmin = lmin if lmin is not None else len(target)
        lmax = lmax if lmax is not None else max(self.words.keys()) + 1
        if stream:  # 这段绷不住了让AI帮我写的，红豆泥私密马赛（鞠躬）
            count = 0
            if shuffle:
                heap = []
                for i in range(lmin, lmax):
                    gen = LinearTrie(self.__fromsize(i), self._wildcard_map).get(target, shuffle=True, stream=True)
                    try:
                        word = next(gen)
                        heapq.heappush(heap, (random.random(), word, gen))
                    except StopIteration:
                        continue

                while heap and (quantity is None or count < quantity):
                    _, word, gen = heapq.heappop(heap)
                    yield word
                    count += 1
                    try:
                        new_word = next(gen)
                        heapq.heappush(heap, (random.random(), new_word, gen))
                    except StopIteration:
                        pass
            else:
                for i in range(lmin, lmax):
                    if quantity is not None and count >= quantity:
                        break
                    for word in LinearTrie(self.__fromsize(i), self._wildcard_map).get(target, shuffle=False, stream=True):
                        yield word
                        count += 1
                        if quantity is not None and count >= quantity:
                            break
            return
        else:
            for i in range(lmin, lmax):
                yield from LinearTrie(self.__fromsize(i), self._wildcard_map).get(target, shuffle=shuffle)

    def set_wildcard(self, wildcard: str, replacement: list[tuple[str, str]]):
        """能看懂吧，我不想写文档了（瘫）"""
        self._wildcard_map[wildcard] = replacement

    def del_wildcard(self, wildcard: str):
        if wildcard in self._wildcard_map:
            del self._wildcard_map[wildcard]


class MapTuple(tuple):
    """带映射的元组。兼容原生tuple，可以用get方法（同dict的）来访问映射值。只能从字典创建。

    主要是给WordList用的:

    >>> import pypinyin
    >>> py = lambda s: " ".join(pypinyin.lazy_pinyin(s))
    >>> words = funcmap(一个词表, py)
    >>> wl = WordList(words)

    然后就可以用WordList类来查拼音什么的
    查到拼音之后:

    >>> next(wl.get("h\\u0000\\u0000 d\\u0000\\u0000 "))
    "hai dai tang"
    >>> words.get("hai dai tang")
    ("海带汤",)

    对于拼音变长问题，可以将声母韵母各自映射为单字符来实现定长
    """

    _val: tuple
    _NODEF = object()

    def __new__(cls, x: dict):
        x = {k: x[k] for k in sorted(x.keys())}
        instance = super().__new__(cls, x.keys())
        instance._val = tuple(x.values())
        return instance

    def get(self, index: Any, default: Any = _NODEF) -> Any:
        i = bisect.bisect_left(self, index)
        if i < len(self) and self[i] == index:
            return self._val[i]
        if default is not MapTuple._NODEF:
            return default
        raise KeyError(index)

    def __repr__(self):
        return f"{super().__repr__()} -> ..."


def funcmap(seq, func: Callable) -> MapTuple:
    """创建函数反查表，从函数值反查输入的"""
    dct = defaultdict(list)
    for item in seq:
        dct[func(item)].append(item)
    return MapTuple({k: tuple(v) for k, v in dct.items()})
