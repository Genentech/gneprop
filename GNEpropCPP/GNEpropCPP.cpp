// Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

// This file provides functions smilesToGraph, smilesToRandomSubgraph,
// and smilesToRandomSubgraphPair for use in Python
// for SMILES parsing, featurization, and random subgraph augmentation.
//
// To build this from scratch and copy the result back to the parent directory:
// git clone https://github.com/pybind/pybind11.git
// mkdir build
// cd build
// cmake -DCMAKE_BUILD_TYPE=Release ..
// time make -j
// cp GNEpropCPP.cpython-39-x86_64-linux-gnu.so ../..
// cd ../..

// Set this to 1 if reording the atoms in each molecule to a canonical ordering
// turns out to be beneficial, to enable that reordering.
// Set it to 0 if the reordering makes no difference, which is expected to be
// more likely, since taking subgraphs partly reorders the atoms anyway.
#define ORDER_ATOMS 0

// Set this to 1 if building a standalone executable to test the
// parsing, featurization, or subgraph handling.  CMakeLists.txt may
// need to be updated.
// Set this to 0 to build a library for use by Python, via pybind11 and torch.
#define STANDALONE 0

// Set this to 1 to collect and print out statistics on the
// atoms and bonds if STANDALONE is 1.
// Set this to 0 if you want to do any performance testing,
// because collecting the statistics might significantly impact performance.
#define REPORT_STATS 0

// Set this to 1 to print out verbose details about the atoms and bonds
// or other info, for use in debugging.
// Set this to 0 if you want to do any performance testing,
// because the logging will probably significantly impact performance.
#define DEBUG_LOGGING 0

// C++ standard library headers
#include <assert.h>
#include <charconv>
#include <memory>
#include <numeric>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// RDKit headers (for SMILES parsing)
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/Canon.h>
#include <GraphMol/new_canon.h>

// Torch tensor headers
#include <ATen/ATen.h>
#include <ATen/Functions.h>

#if STANDALONE
// C++ standard library headers for use by standalone application
#include <chrono>
#else
// PyBind and Torch headers for use by library to be imported by Python
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#endif

namespace {

// These are a bunch of arrays just used for optionally collecting stats
// about the molecules in the data set as they're processed.
// They don't exist if REPORT_STATS is 0.
#if REPORT_STATS
constexpr size_t STATS_NUM_MOL_ATOM_COUNTS = 256;
size_t statsMolAtomCounts[STATS_NUM_MOL_ATOM_COUNTS] = {0};
constexpr size_t STATS_NUM_ELEMENTS = 128;
size_t statsElementCounts[STATS_NUM_ELEMENTS] = {0};
constexpr size_t STATS_NUM_DEGREES = 16;
size_t statsDegreeCounts[STATS_NUM_DEGREES] = {0};
constexpr size_t STATS_NUM_CHARGES = 13;
constexpr size_t STATS_CHARGE_OFFSET = 6;
size_t statsChargeCounts[STATS_NUM_CHARGES] = {0};
constexpr size_t STATS_NUM_CHIRALITIES = 10;
size_t statsChiralityCounts[STATS_NUM_CHIRALITIES] = {0};
constexpr size_t STATS_NUM_HS = 10;
size_t statsHCounts[STATS_NUM_HS] = {0};
constexpr size_t STATS_NUM_HYBRIDIZATIONS = 10;
size_t statsHybridizationCounts[STATS_NUM_HYBRIDIZATIONS] = {0};

size_t statsTotalNumAtoms = 0;
size_t statsAromaticAtomCount = 0;

constexpr size_t STATS_NUM_MOL_BOND_COUNTS = 256;
size_t statsMolBondCounts[STATS_NUM_MOL_BOND_COUNTS] = {0};
constexpr size_t STATS_NUM_BOND_TYPES = 23;
size_t statsBondTypeCounts[STATS_NUM_BOND_TYPES] = {0};
constexpr size_t STATS_NUM_BOND_STEREOS = 7;
size_t statsBondStereoCounts[STATS_NUM_BOND_STEREOS] = {0};

size_t statsTotalNumBonds = 0;
size_t statsConjugatedBondCount = 0;
size_t statsBondInRingCount = 0;
#endif

// Molecule data sets will probably only use a small portion of possible atomic numbers,
// so to avoid having a lot of columns that are entirely zeros, this will map
// atomic numbers to atom feature indices (column numbers) for specific atomic numbers,
// and all other atomic numbers will be mapped to feature index numUniqueElements.
// To add more elements, increase numUniqueElements to the desired number of elements,
// and update the array entries for all supported elements to be unique values starting
// from zero, going up to numUniqueElements-1.
constexpr size_t defaultNumUniqueElements = 22;
size_t numUniqueElements = defaultNumUniqueElements;
std::vector<size_t> atomicNumMap{
	// Atomic number 0 means '*' in SMILES format, for "unknown", which does appear in some molecules.
	// However, it was previously treated as out of bounds, so this continues that treatment.
	defaultNumUniqueElements,
	0, // Although hydrogens are usually implicit, deuterium ('[2H]' in SMILES) is not, and does appear
	defaultNumUniqueElements, // He (2)
	1, // Li (3)
	defaultNumUniqueElements, // Be (4)
	2, // B (5)
	3, // C (6)
	4, // N (7)
	5, // O (8)
	6, // F (9)
	defaultNumUniqueElements, // Ne (10)
	7, // Na (11)
	defaultNumUniqueElements, // Mg (12)
	8, // Al (13)
	9, // Si (14)
	10, // P (15)
	11, // S (16)
	12, // Cl (17)
	defaultNumUniqueElements, // Ar (18)
	defaultNumUniqueElements, // K (19)
	defaultNumUniqueElements, // Ca (20)
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	defaultNumUniqueElements, // Mn (25)
	13, // Fe (26)
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	14, // Cu (29)
	defaultNumUniqueElements, // Zn (30)
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	15, // Br (35)
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	defaultNumUniqueElements, // Zr (40)
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	16, // Ru (44)
	defaultNumUniqueElements, // Rh (45)
	17, // Pd (46)
	18, // Ag (47)
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	19, // Sn (50)
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	20, // I (53)
	defaultNumUniqueElements,
	defaultNumUniqueElements, // Cs (55)
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	defaultNumUniqueElements,
	21  // Pr (59)
};

// If we're not using atomicNumMap, we don't need an array to map atomic numbers
// to atom feature indices.  The feature index will just be the atomic number minus 1
// if it is in the range [1, maxAtomicNumCompatible], and otherwise (including 0 for
// "unknown"), the feature index will be the extra index, maxAtomicNumCompatible.
constexpr size_t maxAtomicNumCompatible = 100;

uint64_t textToUint64(const char*& begin, const char*const end) {
    assert(*begin >= '0' && *begin <= '9');
    uint64_t value = (*begin - '0');
    ++begin;
    while (begin != end && (*begin >= '0' && *begin <= '9')) {
        value = 10*value + (*begin - '0');
        ++begin;
    }
    return value;
}

struct Options {
	// When this is true, feature vectors will be compatible with the original feature vectors,
	// from before adding the C++ library.
	// NOTE: This is global, so do not set this multiple times to different settings for different threads.
	bool compatibleFeatureVectors = false;

	bool useFP16 = false;

	static constexpr const char*const optionsFilename = "GNEpropCPPConfig.txt";
	static constexpr const char*const compatibleName = "compatible";
	static constexpr const size_t compatibleLength = 10;
	static constexpr const char*const precisionName = "precision";
	static constexpr const size_t precisionLength = 9;

	Options() {
		FILE* file = fopen(optionsFilename, "rb");
		if (file == nullptr) {
			// No options file, so stick with defaults.
			return;
		}
		fseek(file, 0, SEEK_END);
		auto fileSize = ftell(file);
		if (fileSize == 0) {
			fclose(file);
			return;
		}
		fseek(file, 0, SEEK_SET);
		std::unique_ptr<char[]> buffer(new char[fileSize]);
		auto countRead = fread(buffer.get(), 1, fileSize, file);
		fclose(file);
		if (countRead != size_t(fileSize)) {
			return;
		}
		const char* begin = buffer.get();
		const char*const end = begin + fileSize;
		while (begin != end) {
			// Skip empty
			while (begin != end && *begin <= ' ') {
				++begin;
			}
			// Find line end
			const char* lineEnd = begin;
			while (lineEnd != end && *lineEnd != '\n' && *lineEnd != '\r') {
				++lineEnd;
			}
			if (lineEnd == begin) {
				// This only happens if we're at the end.
				break;
			}
			if (*begin == '#') {
				// Comment line
				begin = lineEnd;
				continue;
			}
			// Find end of option name
			const char* nameEnd = begin;
			while (nameEnd != lineEnd && (((*nameEnd|0x20) >= 'a' && (*nameEnd|0x20) <= 'z') || (*nameEnd >= '0' && *nameEnd <= '9') || (*nameEnd == '_'))) {
				++nameEnd;
			}
			// Find beginning of value
			const char* valueBegin = nameEnd;
			while (valueBegin != lineEnd && (*valueBegin < '0' || *valueBegin > '9')) {
				++valueBegin;
			}
			if (nameEnd == begin || valueBegin == lineEnd) {
				// No name or no value
				begin = lineEnd;
				continue;
			}
			const size_t nameLength = nameEnd - begin;

			// "compatible"
			if (nameLength == compatibleLength && strncmp(begin, compatibleName, compatibleLength) == 0) {
				const uint64_t value = textToUint64(valueBegin, lineEnd);
				compatibleFeatureVectors = (value != 0);
			}
			// "precision"
			else if (nameLength == precisionLength && strncmp(begin, precisionName, precisionLength) == 0) {
				const uint64_t value = textToUint64(valueBegin, lineEnd);
				useFP16 = (value == 16);
			}

			begin = lineEnd;
		}
	}
};

Options& getOptions() {
	// function-scope static variables are initialized once in a thread-safe way.
	static Options options;
	return options;
}

template<bool fp16>
struct FeatureInfo;

template<>
struct FeatureInfo<true> {
	using Type = uint16_t;
	constexpr static Type ZERO = 0x0000;
	constexpr static Type ONE = 0x3C00;
	constexpr static auto TENSOR_TYPE = c10::ScalarType::Half;

	static Type floatToFeatureType(float f) {
		union {
			float uf;
			uint32_t ui;
		} u;
		u.uf = f;
		uint32_t i = u.ui;
		const uint16_t sign = uint16_t((i & 0x80000000) >> 16);
		i = (i & 0x7FFFFFFF);
		// If it rounds down to a finite number
		if (i < (((15 + 127)<<23) | (0x7FF << 12))) {
			const int exponent = (i >> 23) - 127;
			if (exponent < -14) {
				// Subnormal number
				// TODO: Implement this properly if needed; currently just rounding to zero.
				return sign;
			}

			// Normal number
			const uint32_t mantissa32 = (i & 0x7FFFFF);
			const uint32_t mantissa16 = mantissa32 >> (23-10);
			uint16_t out = uint16_t(((exponent + 15) << 23) | mantissa16);
			if ((mantissa32 & 0x1FFF) == 0x1000 && (mantissa16 & 1) != 0) {
				++out;
			}
			return out | sign;
		}
		if (i <= 0x7F800000) {
			// Infinity
			return 0x7C00 | sign;
		}

		// NaN
		return 0x7C01 | sign | ((i & 0x00400000) >> 13);
	}
};

template<>
struct FeatureInfo<false> {
	using Type = float;
	constexpr static Type ZERO = 0.0f;
	constexpr static Type ONE = 1.0f;
	constexpr static auto TENSOR_TYPE = c10::ScalarType::Float;

	static constexpr Type floatToFeatureType(float f) {
		return f;
	}
};

// These variables are just for determining the first atom feature index (offset)
// of some type, and the number of feature indices for it.  The +1's are
// extra feature indices for if some value is outside the explicitly supported range.
constexpr size_t numDegrees = 6;
constexpr size_t numChargeOptions = 5;
constexpr size_t numChiralTags = 4;
constexpr size_t numHsLimit = 5;
constexpr size_t numHybridizations = 5;
constexpr size_t numFeaturesAfterElements =
	(numDegrees + 1) +
	(numChargeOptions + 1) +
	(numChiralTags + 1) +
	(numHsLimit + 1) +
	(numHybridizations + 1) +
	2;

template<typename FeatureType>
struct AtomFeaturesAfterElements {
	FeatureType degreeFeatures[numDegrees + 1];
	FeatureType chargeFeatures[numChargeOptions + 1];
	FeatureType chiralFeatures[numChiralTags + 1];
	FeatureType HsFeatures[numHsLimit + 1];
	FeatureType hybridizationFeatures[numHybridizations + 1];
	FeatureType aromaticFeature;
	FeatureType massFeature;
};

// This is a map from RDKit HybridizationType enum values to atom hybridization
// feature indices in the range [0,5], matching the mapping from the original
// GNEprop Python code.
// HybridizationType.SP (2) -> 0
// HybridizationType.SP2 (3) -> 1
// HybridizationType.SP3 (4) -> 2
// HybridizationType.SP3D (6) -> 3
// HybridizationType.SP3D2 (7) -> 4
constexpr size_t hybridizationMapSize = 9;
constexpr size_t hybridizationMap[hybridizationMapSize] = {
	numHybridizations, // UNSPECIFIED
	numHybridizations, // S
	0,                 // SP
	1,                 // SP2
	2,                 // SP3
	numHybridizations, // SP2D
	3,                 // SP3D
	4,                 // SP3D2
	numHybridizations  // OTHER
};

// This is a map from RDKit BondType enum values to bond feature indices,
// with the numBondTypes entries not being associated with a feature
// index (the feature will just be zero), so the valid range is [0,3].
// The BondType enum is actually longer than this, but ones after AROMATIC
// will just be skipped (feature zeroed).
// BondType.SINGLE (1) -> 0
// BondType.DOUBLE (2) -> 1
// BondType.TRIPLE (3) -> 2
// BondType.AROMATIC (12) -> 3
constexpr size_t numBondTypes = 4;
constexpr size_t bondTypeMapSize = 13;
constexpr size_t bondTypeMap[bondTypeMapSize] = {
	numBondTypes, // UNSPECIFIED
	0,              // SINGLE
	1,              // DOUBLE
	2,              // TRIPLE
	numBondTypes, // QUADRUPLE
	numBondTypes, // QUINTUPLE
	numBondTypes, // HEXTUPLE
	numBondTypes, // ONEANDAHALF
	numBondTypes, // TWOANDAHALF
	numBondTypes, // THREEANDAHALF
	numBondTypes, // FOURANDAHALF
	numBondTypes, // FIVEANDAHALF
	3               // AROMATIC
};

// These variables are just for predetermining the first bond feature index (offset)
// of some type, and the number of feature indices for it.
constexpr size_t bondOtherOffset = numBondTypes;
constexpr size_t bondNumOther = 2;

// TODO: This is a trivial map, so consider removing it or removing BondStereo
// values that don't occur in the data set.
// BondStereo.STEREONONE (0) -> 0
// BondStereo.STEREOANY (1) -> 1
// BondStereo.STEREOZ (2) -> 2
// BondStereo.STEREOE (3) -> 3
// BondStereo.STEREOCIS (4) -> 4
// BondStereo.STEREOTRANS (5) -> 5
constexpr size_t bondStereoOffset = bondOtherOffset + bondNumOther;
constexpr size_t bondNumStereo = 6;
constexpr size_t bondStereoMapSize = 6;
constexpr size_t bondStereoMap[bondStereoMapSize] = {
	0,
	1,
	2,
	3,
	4,
	5
};

constexpr size_t numTotalBondFeat = bondStereoOffset + bondNumStereo;

// This is just a wrapper of the random number generator state to make
// working with the random number generator simpler.
struct RNG {
	uint64_t state[4];

	RNG() {
		// TODO: Really, the random initialization should be coming from the process using these,
		//       to ensure that they're unique to all workers.
		// These are just arbitrary randomly-generated numbers.
		state[0] = 0x85115e237b20d394;
		state[1] = 0x17b92d8b1c609881;
		state[2] = 0x4c0aa0e54d283279;
		state[3] = 0xb5c3e1c50fae38b2;
		state[1] ^= std::hash<std::thread::id>()(std::this_thread::get_id());
		state[0] ^= uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());

		// Mix it up a little
		random();
		random();
	}

	RNG(uint64_t seed) {
		state[0] = 0x4c0aa0e54d283279 ^ seed;
		state[1] = 0x85115e237b20d394 ^ seed;
		state[2] = 0x17b92d8b1c609881 ^ seed;
		state[3] = 0xb5c3e1c50fae38b2 ^ seed;

		// Mix it up a little
		random();
		random();
	}

	static constexpr uint64_t rotateLeft(const uint64_t x, int k) {
		return (x << k) | (x >> (64 - k));
	}

	// Call this to get the next random 64-bit unsigned integer.
	uint64_t random() {
		// xoshiro256** public domain (CC0) from https://prng.di.unimi.it/xoshiro256starstar.c
		const uint64_t result = rotateLeft(state[1] * 5, 7) * 9;

		const uint64_t t = state[1] << 17;

		state[2] ^= state[0];
		state[3] ^= state[1];
		state[1] ^= state[2];
		state[0] ^= state[3];

		state[2] ^= t;

		state[3] = rotateLeft(state[3], 45);

		return result;
	}
};

RNG& getRNG() {
	// TODO: Initialize the random number generator based on info from the calling application,
	// to ensure that no two workers have the same random seed, though this does make
	// an effort to greatly reduce the chance of that.  Since different workers
	// see different molecules, and hopefully don't get restarted on every epoch,
	// this might not be an issue, but it tends to be best practice to avoid the risk.
    thread_local RNG rng;
    return rng;
}
} // End of anonymous namespace

void configCPPOptions(bool useCompatibleFeatureVectors, const std::vector<size_t>& newAtomicNumMap, size_t newNumUniqueElements, bool newUseFP16 = false) {
	Options& options = getOptions();
	options.compatibleFeatureVectors = useCompatibleFeatureVectors;
	if (newAtomicNumMap.size() > 0 && newNumUniqueElements < 128 && newNumUniqueElements > 0) {
		// Validate the new atomic number map, just in case.
		bool valid = true;
		uint64_t alreadyPresent[2] = {0, 0};
		for (size_t i = 0; i < atomicNumMap.size(); ++i) {
			size_t index = atomicNumMap[i];
			if (index > newNumUniqueElements) {
				valid = false;
				break;
			}
			if (index < newNumUniqueElements) {
				// Each valid index should only appear once.
				// If this is ever too strict of a requirement, feel free to remove it.
				const uint64_t mask = 1ULL << (index & 0x3F);
				if (alreadyPresent[index >> 6] & mask) {
					valid = false;
					break;
				}
				alreadyPresent[index >> 6] |= mask;
			}
		}
		if (valid) {
			atomicNumMap = newAtomicNumMap;
			numUniqueElements = newNumUniqueElements;
		}
		else {
			printf("WARNING: Invalid newAtomicNumMap in configCPPOptions; ignoring.\n");
			fflush(stdout);
			atomicNumMap.resize(0);
		}
	}
	options.useFP16 = newUseFP16;
}

namespace {

// The data needed about an atom
struct CompactAtom {
	uint8_t atomicNum;
	uint8_t totalDegree;
	int8_t formalCharge;
	uint8_t chiralTag;
	uint8_t totalNumHs;
	uint8_t hybridization;
	bool isAromatic;
	float mass;
};

// The data needed about a bond
struct CompactBond {
	uint8_t bondType;
	bool isConjugated;
	bool isInRing;
	uint8_t stereo;
	uint32_t beginAtomIdx;
	uint32_t endAtomIdx;
};

// Data representing a molecule before featurization or augmentation
struct GraphData {
	const size_t numAtoms;
	std::unique_ptr<CompactAtom[]> atoms;
	const size_t numBonds;
	std::unique_ptr<CompactBond[]> bonds;
};

// RDKit::SmilesToMol uses std::string, so until we replace it, lets use std::string here.
// ("const char*" could avoid an extra allocation, if we do eventually replace use of SmilesToMol.)
GraphData readGraph(const std::string& smilesString) {
	// Parse SMILES string with default options
	RDKit::SmilesParserParams params;
	std::unique_ptr<RDKit::ROMol> mol{ RDKit::SmilesToMol(smilesString, params) };

	const size_t numAtoms = mol->getNumAtoms();
	const size_t numBonds = mol->getNumBonds();
#if DEBUG_LOGGING
	printf("# atoms = %zu\n# bonds = %zu\n", numAtoms, numBonds);
#endif
#if REPORT_STATS
	++statsMolAtomCounts[(numAtoms >= STATS_NUM_MOL_ATOM_COUNTS) ? (STATS_NUM_MOL_ATOM_COUNTS-1) : numAtoms];
	++statsMolBondCounts[(numBonds >= STATS_NUM_MOL_BOND_COUNTS) ? (STATS_NUM_MOL_BOND_COUNTS-1) : numBonds];
	statsTotalNumAtoms += numAtoms;
	statsTotalNumBonds += numBonds;
#endif

#if ORDER_ATOMS
	// Determine a canonical ordering of the atoms, if desired.
	std::vector<unsigned int> atomOrder;
	atomOrder.reserve(numAtoms);
	RDKit::Canon::rankMolAtoms(*mol, atomOrder);
	assert(atomOrder.size() == numAtoms);
#endif

	// Allocate an array of atom data, and fill it from the RDKit atom data.
	std::unique_ptr<CompactAtom[]> atoms(new CompactAtom[numAtoms]);
	for (size_t atomIdx = 0; atomIdx < numAtoms; ++atomIdx) {
		const RDKit::Atom* const atom = mol->getAtomWithIdx(atomIdx);
		auto atomicNum = atom->getAtomicNum();
		auto totalDegree = atom->getTotalDegree();
		auto formalCharge = atom->getFormalCharge();
		const RDKit::Atom::ChiralType chiralType = atom->getChiralTag();
		auto totalNumHs = atom->getTotalNumHs();
		const RDKit::Atom::HybridizationType hybridization = atom->getHybridization();

		const bool isAromatic = atom->getIsAromatic();
#if REPORT_STATS
		++statsElementCounts[(atomicNum < 0 || atomicNum >= STATS_NUM_ELEMENTS) ? (STATS_NUM_ELEMENTS-1) : atomicNum];
		++statsDegreeCounts[(totalDegree < 0 || totalDegree >= STATS_NUM_DEGREES) ? (STATS_NUM_DEGREES-1) : totalDegree];
		size_t formalChargeIndex = formalCharge + int(STATS_CHARGE_OFFSET);
		if (formalCharge < -int(STATS_CHARGE_OFFSET)) {
			formalChargeIndex = 0;
		}
		else if (formalCharge > int(STATS_CHARGE_OFFSET)) {
			formalChargeIndex = STATS_NUM_CHARGES-1;
		}

		++statsChargeCounts[formalChargeIndex];
		++statsChiralityCounts[(size_t(chiralType) >= STATS_NUM_CHIRALITIES) ? (STATS_NUM_CHIRALITIES-1) : size_t(chiralType)];
		++statsHCounts[(totalNumHs < 0 || totalNumHs >= STATS_NUM_HS) ? (STATS_NUM_HS-1) : totalNumHs];
		++statsHybridizationCounts[(size_t(hybridization) >= STATS_NUM_HYBRIDIZATIONS) ? (STATS_NUM_HYBRIDIZATIONS-1) : size_t(hybridization)];
		statsAromaticAtomCount += (isAromatic ? 1 : 0);
#endif
		const double mass = atom->getMass();

#if ORDER_ATOMS
		const size_t destAtomIdx = atomOrder[atomIdx];
#else
		const size_t destAtomIdx = atomIdx;
#endif
		atoms[destAtomIdx] = CompactAtom{
			uint8_t(atomicNum),
			uint8_t(totalDegree),
			int8_t(formalCharge),
			uint8_t(chiralType),
			uint8_t(totalNumHs),
			uint8_t(hybridization),
			isAromatic,
			float(mass)
		};
#if DEBUG_LOGGING
		printf(
			"atom[%zu] = {%zu, %u, %d, %u, %u, %u, %s, %f}\n",
			destAtomIdx,
			int(atomicNum),
			int(totalDegree),
			int(formalCharge),
			int(chiralType),
			int(totalNumHs),
			int(hybridization),
			isAromatic ? "true" : "false",
			mass
		);
#endif
	}

	// Allocate an array of bond data, and fill it from the RDKit bond data.
	std::unique_ptr<CompactBond[]> bonds(new CompactBond[numBonds]);
	const RDKit::RingInfo*const ringInfo = mol->getRingInfo();
	for (size_t bondIdx = 0; bondIdx < numBonds; ++bondIdx) {
		const RDKit::Bond*const bond = mol->getBondWithIdx(bondIdx);
		const RDKit::Bond::BondType bondType = bond->getBondType();
		const bool isConjugated = bond->getIsConjugated();
		// TODO: Verify that it's the same index as bond->getIdx()
		const bool isInRing = (ringInfo->numBondRings(bondIdx) != 0);
		const RDKit::Bond::BondStereo stereo = bond->getStereo();

#if REPORT_STATS
		++statsBondTypeCounts[(size_t(bondType) >= STATS_NUM_BOND_TYPES) ? (STATS_NUM_BOND_TYPES-1) : size_t(bondType)];
		++statsBondStereoCounts[(size_t(stereo) >= STATS_NUM_BOND_STEREOS) ? (STATS_NUM_BOND_STEREOS-1) : size_t(stereo)];
		statsConjugatedBondCount += (isConjugated ? 1 : 0);
		statsBondInRingCount += (isInRing ? 1 : 0);
#endif

		auto beginAtomIdx = bond->getBeginAtomIdx();
		auto endAtomIdx = bond->getEndAtomIdx();
#if ORDER_ATOMS
		beginAtomIdx = atomOrder[beginAtomIdx];
		endAtomIdx = atomOrder[endAtomIdx];
#endif
		bonds[bondIdx] = CompactBond{
			uint8_t(bondType),
			isConjugated,
			isInRing,
			uint8_t(stereo),
			beginAtomIdx,
			endAtomIdx
		};
#if DEBUG_LOGGING
		printf(
			"bond[%zu] = {%u, %s, %s, %u, {%u, %u}}\n",
			bondIdx,
			int(bondType),
			isConjugated ? "true" : "false",
			isInRing ? "true" : "false",
			int(stereo),
			beginAtomIdx,
			endAtomIdx
		);
#endif
	}

	// Return a GraphData structure, taking ownership of the atom and bond data arrays.
	return GraphData{ numAtoms, std::move(atoms), numBonds, std::move(bonds) };
}

// This is a structure for managing the adjacency data (CSR format) for use by randomSubgraph.
struct NeighbourData {
	// This owns the data of all 3 arrays, which are actually a single, contiguous allocation.
	std::unique_ptr<size_t[]> deleter;

	// This is an array of indices into the other two arrays, indicating where
	// each atom's neighbours start, including the first entry being 0 for the start of
	// atom 0, and the numAtoms entry being 2*numBonds (2x because each bond is on 2 atoms),
	// so there are numAtoms+1 entries.  The number of neighbours of an atom i is
	// neighbourStarts[i+1]-neighbourStarts[i]
	const size_t* neighbourStarts;

	// The neighbour atom for each bond, with each atom having an entry for each of
	// its neighbours, so each bond occurs twice.
	const size_t* neighbours;

	// This is in the same order as neighbours, but indicates the index of the bond.
	// Each bond occurs twice, so each number occurs twice.
	const size_t* bondIndices;
};

// Construct a NeighbourData structure representing the molecule's graph in CSR format.
NeighbourData constructNeighbours(const GraphData& graph) {
	const size_t numAtoms = graph.numAtoms;
	const size_t numBonds = graph.numBonds;
	// Do a single allocation for all 3 arrays.
	std::unique_ptr<size_t[]> deleter(new size_t[numAtoms + 1 + 4 * numBonds]);

	size_t* neighbourStarts = deleter.get();
	for (size_t i = 0; i <= numAtoms; ++i) {
		neighbourStarts[i] = 0;
	}

	// First, get atom neighbour counts
	const CompactBond* const bonds = graph.bonds.get();
	for (size_t i = 0; i < numBonds; ++i) {
		size_t a = bonds[i].beginAtomIdx;
		size_t b = bonds[i].endAtomIdx;
		// NOTE: +1 is because first entry will stay zero.
		++neighbourStarts[a + 1];
		++neighbourStarts[b + 1];
	}

	// Find the starts by partial-summing the neighbour counts.
	// NOTE: +1 is because first entry will stay zero.
	std::partial_sum(neighbourStarts + 1, neighbourStarts + 1 + numAtoms, neighbourStarts + 1);

	// Fill in the neighbours and bondIndices arrays.
	size_t* neighbours = neighbourStarts + numAtoms + 1;
	size_t* bondIndices = neighbours + 2*numBonds;
	for (size_t i = 0; i < numBonds; ++i) {
		size_t a = bonds[i].beginAtomIdx;
		size_t b = bonds[i].endAtomIdx;

		size_t ai = neighbourStarts[a];
		neighbours[ai] = b;
		bondIndices[ai] = i;
		++neighbourStarts[a];

		size_t bi = neighbourStarts[b];
		neighbours[bi] = a;
		bondIndices[bi] = i;
		++neighbourStarts[b];
	}

	// Shift neighbourStarts forward one after incrementing it.
	size_t previous = 0;
	for (size_t i = 0; i < numAtoms; ++i) {
		size_t next = neighbourStarts[i];
		neighbourStarts[i] = previous;
		previous = next;
	}

	// NeighbourData takes ownership of the memory.
	return NeighbourData{ std::move(deleter), neighbourStarts, neighbours, bondIndices };
}

// A representation of the parts of the graph kept in a particular subgraph.
struct Subgraph {
	size_t numNodesKept;
	size_t* nodesKept;
	size_t numEdgesKept;
	size_t* edgesKept;
	std::unique_ptr<uint64_t[]> edgeIndex;
};

// From a molecule's graph, select a subgraph with a random connected
// subgraph of numToRemove atoms removed.
// visited must be length numNodes
// neighbourStarts must be length numNodes+1
// neighbours must be length neighbourStarts[numNodes]
// bondIndices must be length neighbourStarts[numNodes]
// neighbourQueue must be length at least numNodes (it also gets used for a node index map)
// nodesToKeep must be length numNodes
// edgesToKeep must be length neighbourStarts[numNodes]
Subgraph randomSubgraph(
	RNG& rng,
	const size_t numNodes,
	uint8_t*const visited,
	const size_t*const neighbourStarts,
	const size_t*const neighbours,
	const size_t*const bondIndices,
	size_t*const neighbourQueue,
	const size_t numToRemove,
	size_t*const nodesToKeep,
	size_t*const edgesToKeep
) {
	// Start with all nodes unvisited (0)
	for (size_t i = 0; i < numNodes; ++i) {
		visited[i] = 0;
	}

	assert(numToRemove >= 1);

	// Select a random first node and mark it as visited and selected (2)
	size_t s =  rng.random() % numNodes;
	visited[s] = 2;
	size_t sampleSize = 1;

	// Add all of its neighbours to the neighbourQueue
	size_t neighbourQueueSize = 0;
	size_t neighbourStart = neighbourStarts[s];
	size_t neighbourEnd = neighbourStarts[s+1];
	for (size_t i = neighbourStart; i < neighbourEnd; ++i) {
		const size_t neighbour = neighbours[i];
		if (visited[neighbour] == 0) {
			// Visited but not selected (1)
			visited[neighbour] = 1;
			neighbourQueue[neighbourQueueSize] = neighbour;
			++neighbourQueueSize;
		}
	}

	if (numToRemove > 1) {
		// Early exit if we exhaust a whole connected component, which can happen
		// with "dot bonds" in SMILES that aren't actually bonds.
		while (neighbourQueueSize != 0) {
			// Select a random neighbour in the queue to select (value 2 in visited)
			const size_t si = rng.random() % neighbourQueueSize;
			const size_t s = neighbourQueue[si];
			visited[s] = 2;
			++sampleSize;
			if (sampleSize == numToRemove) {
				// We have enough nodes in the sample.
				break;
			}
			// Remove the neighbour from the queue by writing the last entry over top of it.
			--neighbourQueueSize;
			neighbourQueue[si] = neighbourQueue[neighbourQueueSize];

			// Add all unvisited neighbours of the selected node to neighbourQueue
			const size_t neighbourStart = neighbourStarts[s];
			const size_t neighbourEnd = neighbourStarts[s + 1];
			for (size_t i = neighbourStart; i < neighbourEnd; ++i) {
				const size_t neighbour = neighbours[i];
				if (visited[neighbour] == 0) {
					// Visited but not selected (1)
					visited[neighbour] = 1;
					neighbourQueue[neighbourQueueSize] = neighbour;
					++neighbourQueueSize;
				}
			}
		}
	}

	// Create a map from old nodes to new nodes (nodeMap)
	// and a map from new nodes to old nodes (nodesToKeep).
	// Reuse the memory for the neighbourQueue, since it's no longer needed.
	size_t*const nodeMap = neighbourQueue;
	//size_t* const nodesToKeep = sampleToRemove + sampleSize;
	size_t numNodesToKeep = 0;
	for (size_t nodei = 0; nodei < numNodes; ++nodei) {
		if (visited[nodei] != 2) {
			// Node not selected to be removed, so it's kept
			nodesToKeep[numNodesToKeep] = nodei;
			nodeMap[nodei] = numNodesToKeep;
			++numNodesToKeep;
		}
	}

	// Determine the edges to keep (both of its nodes must be kept)
	size_t numEdgesToKeep = 0;
	for (size_t nodei = 0; nodei < numNodes; ++nodei) {
		if (visited[nodei] == 2) {
			continue;
		}
		const size_t neighbourStart = neighbourStarts[nodei];
		const size_t neighbourEnd = neighbourStarts[nodei + 1];
		for (size_t edgei = neighbourStart; edgei < neighbourEnd; ++edgei) {
			const size_t neighbour = neighbours[edgei];
			if (visited[neighbour] != 2) {
				edgesToKeep[numEdgesToKeep] = bondIndices[edgei];
				++numEdgesToKeep;
			}
		}
	}

	// Allocate an array and remap the nodes for each edge
	std::unique_ptr<uint64_t[]> edgeIndex(new uint64_t[2 * numEdgesToKeep]);
	uint64_t* edgeIndex0 = edgeIndex.get();
	uint64_t* edgeIndex1 = edgeIndex0 + numEdgesToKeep;
	numEdgesToKeep = 0;
	for (size_t nodei = 0; nodei < numNodes; ++nodei) {
		if (visited[nodei] == 2) {
			continue;
		}
		const size_t newNodei = nodeMap[nodei];
		const size_t neighbourStart = neighbourStarts[nodei];
		const size_t neighbourEnd = neighbourStarts[nodei + 1];
		for (size_t edgei = neighbourStart; edgei < neighbourEnd; ++edgei) {
			const size_t neighbour = neighbours[edgei];
			if (visited[neighbour] != 2) {
				edgeIndex0[numEdgesToKeep] = newNodei;
				edgeIndex1[numEdgesToKeep] = nodeMap[neighbour];
				++numEdgesToKeep;
			}
		}
	}

	// Return a Subgraph, taking ownership of edgeIndex array.
	return Subgraph{ numNodesToKeep, nodesToKeep, numEdgesToKeep, edgesToKeep, std::move(edgeIndex) };
}

// Feature data tensors and dimensions
template<typename FeatureType>
struct Features {
	size_t nodeFeatDim[2];
	std::unique_ptr<FeatureType[]> nodeFeat;
	size_t edgeFeatDim[2];
	std::unique_ptr<FeatureType[]> edgeFeat;
	size_t edgeIndexDim[2];
	std::unique_ptr<uint64_t[]> edgeIndex;
};

// Given a molecule graph and an indication of what's being kept of it
// in a subgraph, construct the node and edge features.
template<bool fp16>
Features<typename FeatureInfo<fp16>::Type> createFeatures(const GraphData& graph, const bool compatibleFeatureVectors, Subgraph* subgraph = nullptr) {
	using Info = FeatureInfo<fp16>;
	using Type = typename Info::Type;
	constexpr auto ZERO = Info::ZERO;
	constexpr auto ONE = Info::ONE;
	Features<Type> output;

	const size_t atomicNumMapSize = atomicNumMap.size();
	const bool useSparseAtoms = !compatibleFeatureVectors && atomicNumMapSize != 0;
	const size_t elementFeatureSize = useSparseAtoms ? (numUniqueElements + 1) : (maxAtomicNumCompatible + 1);

	// Fill in the tensor dimensions.
	const size_t numNodes = (subgraph != nullptr) ? subgraph->numNodesKept : graph.numAtoms;
	const size_t numEdges = (subgraph != nullptr) ? subgraph->numEdgesKept : 2*graph.numBonds;
	output.nodeFeatDim[0] = numNodes;
	output.nodeFeatDim[1] = elementFeatureSize + numFeaturesAfterElements;
	output.edgeFeatDim[0] = numEdges;
	output.edgeFeatDim[1] = numTotalBondFeat;
	output.edgeIndexDim[0] = 2;
	output.edgeIndexDim[1] = numEdges;

	// Allocate the memory for the node and edge features.
	const size_t nodeFeatTotalSize = output.nodeFeatDim[0] * output.nodeFeatDim[1];
	const size_t edgeFeatTotalSize = output.edgeFeatDim[0] * output.edgeFeatDim[1];
	output.nodeFeat.reset(new Type[nodeFeatTotalSize]);
	output.edgeFeat.reset(new Type[edgeFeatTotalSize]);

	if (subgraph != nullptr) {
		// Take ownership of edgeIndex from subgraph
		output.edgeIndex = std::move(subgraph->edgeIndex);
	}
	else {
		// Create edgeIndex from scratch
		std::unique_ptr<uint64_t[]> edgeIndex(new uint64_t[2*numEdges]);
		for (size_t i = 0; i < graph.numBonds; ++i) {
			edgeIndex[2*i] = graph.bonds[i].beginAtomIdx;
			edgeIndex[2*i+1] = graph.bonds[i].endAtomIdx;
			edgeIndex[2*i + 2*graph.numBonds] = graph.bonds[i].endAtomIdx;
			edgeIndex[2*i+1 + 2*graph.numBonds] = graph.bonds[i].beginAtomIdx;
		}
		output.edgeIndex = std::move(edgeIndex);
	}

	// Fill in nodeFeat

	// nodeFeat will be mostly zero, so zero it out with memset first.
	memset(output.nodeFeat.get(), 0, nodeFeatTotalSize * sizeof(Type));

	Type* rowStart = output.nodeFeat.get();
	for (size_t i = 0; i < numNodes; ++i, rowStart += output.nodeFeatDim[1]) {
		const size_t atomIndex = (subgraph != nullptr) ? subgraph->nodesKept[i] : i;
		const CompactAtom& atom = graph.atoms[atomIndex];

		// Map atomic number, degree, charge, chirality, # of Hs, and hybridization
		// to feature indices, and set the corresponding index to 1.0.
		// This is the one-hot encoding used originally, though possibly with
		// some of the feature indices in a different order, and if useSparseAtoms
		// is true, some indices (columns) that were entirely zero have been removed.
		size_t index = atom.atomicNum;
		if (useSparseAtoms) {
			index = (index >= atomicNumMapSize) ? numUniqueElements : atomicNumMap[index];
		}
		else {
			--index;
			if (index >= maxAtomicNumCompatible) {
				index = maxAtomicNumCompatible;
			}
		}
		rowStart[index] = ONE;

		index = atom.totalDegree;
		if (index >= numDegrees) {
			index = numDegrees;
		}

		static_assert(sizeof(AtomFeaturesAfterElements<Type>) == sizeof(Type)*numFeaturesAfterElements);

		AtomFeaturesAfterElements<Type>*const atomFeatures = reinterpret_cast<AtomFeaturesAfterElements<Type>*>(rowStart + elementFeatureSize);
		atomFeatures->degreeFeatures[index] = ONE;

		index = atom.formalCharge + 2;
		if (index >= numChargeOptions) {
			index = numChargeOptions;
		}

		if (compatibleFeatureVectors) {
			// Map charges back to the feature indices used in the original order:
			// -1, -2, 1, 2, 0, other
			constexpr size_t formalChargeIndexMap[6] = {
				1,
				0,
				4,
				2,
				3,
				5
			};
			index = formalChargeIndexMap[index];
		}

		atomFeatures->chargeFeatures[index] = ONE;

		index = atom.chiralTag;
		if (index >= numChiralTags) {
			index = numChiralTags;
		}
		atomFeatures->chiralFeatures[index] = ONE;

		index = atom.totalNumHs;
		if (index >= numHsLimit) {
			index = numHsLimit;
		}
		atomFeatures->HsFeatures[index] = ONE;

		index = atom.hybridization;
		index = (index >= hybridizationMapSize) ? numHybridizations : hybridizationMap[index];
		atomFeatures->hybridizationFeatures[index] = ONE;

		// isAromatic and mass are already values that only need a single
		// feature index to represent them.
		atomFeatures->aromaticFeature = atom.isAromatic ? ONE : ZERO;
		atomFeatures->massFeature = Info::floatToFeatureType(atom.mass * 0.01f);
	}

	// Fill in edgeFeat

	// edgeFeat will be mostly zero, so zero it out with memset first.
	memset(output.edgeFeat.get(), 0, edgeFeatTotalSize * sizeof(Type));

	rowStart = output.edgeFeat.get();
	for (size_t i = 0; i < numEdges; ++i, rowStart += output.edgeFeatDim[1]) {
		const size_t bondIndex = (subgraph != nullptr) ? subgraph->edgesKept[i] : (i/2);
		const CompactBond& bond = graph.bonds[bondIndex];

		// Map bond type and bond stereochemistry to feature indices,
		// and set the corresponding index to 1.0, if it's a valid index.
		// This is the one-hot encoding used originally.
		// isConjugated and isInRing only need a single feature index.
		size_t index = bond.bondType;
		if (index < bondTypeMapSize) {
			index = bondTypeMap[index];
			if (index < numBondTypes) {
				rowStart[index] = ONE;
			}
		}

		rowStart[bondOtherOffset + 0] = bond.isConjugated ? ONE : ZERO;
		rowStart[bondOtherOffset + 1] = bond.isInRing ? ONE : ZERO;

		index = bond.stereo;
		if (index < bondStereoMapSize) {
			index = bondStereoMap[index];
			if (index < bondNumStereo) {
				rowStart[bondStereoOffset + index] = ONE;
			}
		}
	}

	// Return the feature data.  The caller receives ownership of the data.
	return output;
}

} // End of anonymous namespace

// This is just a function to provide to torch, so that we don't have to copy
// the tensor data to put it in a torch tensor, and torch can delete the data
// when it's no longer needed.
template<typename T>
void deleter(void* p) {
	delete [] (T*)p;
}

template<bool fp16>
void createFeatureTensors(const GraphData& graph, const bool compatibleFeatureVectors, Subgraph* subgraph, at::Tensor& nodeFeat, at::Tensor& edgeFeat, at::Tensor& edgeIndex) {
	// Create the features for this graph.
	auto features = createFeatures<fp16>(graph, compatibleFeatureVectors, subgraph);

	using Info = FeatureInfo<fp16>;
	using Type = typename Info::Type;

	// Set up the torch tensors to return, taking ownership of the feature
	// data that's already been initialized by createFeatures.
	nodeFeat = at::from_blob(
		features.nodeFeat.release(),
		{int64_t(features.nodeFeatDim[0]), int64_t(features.nodeFeatDim[1])},
		deleter<Type>, c10::TensorOptions(Info::TENSOR_TYPE));
	edgeFeat = at::from_blob(
		features.edgeFeat.release(),
		{int64_t(features.edgeFeatDim[0]), int64_t(features.edgeFeatDim[1])},
		deleter<Type>, c10::TensorOptions(Info::TENSOR_TYPE));

	edgeIndex = at::from_blob(
		features.edgeIndex.release(),
		{int64_t(features.edgeIndexDim[0]), int64_t(features.edgeIndexDim[1])},
		deleter<uint64_t>, c10::TensorOptions(c10::ScalarType::Long));
}

// This is the function exported to Python with pybind11.
// It receives a text string representing a molecule in SMILES format,
// and returns a vector contining 3 tensors representing the full graph of the molecule.
// The 3 tensors for each subgraph are: node features, edge features, edge index.
std::vector<at::Tensor>
smilesToGraph(const std::string& smilesString) {
	// Get the molecule graph data in a useful format
	GraphData graph = readGraph(smilesString);

	Options& options = getOptions();
	const bool compatibleFeatureVectors = options.compatibleFeatureVectors;
	const bool useFP16 = options.useFP16;

	// Create the features for this graph.
	at::Tensor nodeFeat;
	at::Tensor edgeFeat;
	at::Tensor edgeIndex;
	if (useFP16) {
		createFeatureTensors<true>(graph, compatibleFeatureVectors, nullptr, nodeFeat, edgeFeat, edgeIndex);
	}
	else {
		createFeatureTensors<false>(graph, compatibleFeatureVectors, nullptr, nodeFeat, edgeFeat, edgeIndex);
	}

	// Return all 3 tensors in a vector
	return std::vector<at::Tensor>{
		std::move(nodeFeat),
		std::move(edgeFeat),
		std::move(edgeIndex)
	};
}

// This is the function exported to Python with pybind11.
// It receives a text string representing a molecule in SMILES format,
// and returns a vector contining 3 tensors representing a subgraph.
// The 3 tensors for each subgraph are: node features, edge features, edge index.
std::vector<at::Tensor>
smilesToRandomSubgraph(const std::string& smilesString, const double fractionNodesToRemove) {
	// Get the molecule graph data in a useful format
	GraphData graph = readGraph(smilesString);

	// Set up the graph representation in CSR (compressed sparse row) format,
	// for faster graph traversal in randomSubgraph.
	NeighbourData neighbourData = constructNeighbours(graph);

	// Allocate memory for use by randomSubgraph, reusing memory and using a single
	// allocation for multiple arrays, to reduce the number of allocations.
	// NOTE: The +1 is for consistency with the previous code that also removed
	//       one more than the floor.
	size_t numNodesToRemove = size_t(graph.numAtoms * fractionNodesToRemove) + 1;
	std::unique_ptr<uint8_t[]> visited(new uint8_t[graph.numAtoms]);
	std::unique_ptr<size_t[]> neighbourQueue(new size_t[graph.numAtoms + graph.numAtoms + 2*graph.numBonds]);
	size_t*const nodesToKeep = neighbourQueue.get() + graph.numAtoms;
	size_t*const edgesToKeep = nodesToKeep + graph.numAtoms;

	RNG& rng = getRNG();

	// Generate a random subgraph
	Subgraph subgraph = randomSubgraph(
		rng,
		graph.numAtoms,
		visited.get(),
		neighbourData.neighbourStarts,
		neighbourData.neighbours,
		neighbourData.bondIndices,
		neighbourQueue.get(),
		numNodesToRemove,
		nodesToKeep,
		edgesToKeep
	);

	Options& options = getOptions();
	const bool compatibleFeatureVectors = options.compatibleFeatureVectors;
	const bool useFP16 = options.useFP16;

	// Create the features for this graph.
	at::Tensor nodeFeat;
	at::Tensor edgeFeat;
	at::Tensor edgeIndex;
	if (useFP16) {
		createFeatureTensors<true>(graph, compatibleFeatureVectors, &subgraph, nodeFeat, edgeFeat, edgeIndex);
	}
	else {
		createFeatureTensors<false>(graph, compatibleFeatureVectors, &subgraph, nodeFeat, edgeFeat, edgeIndex);
	}

	// Return all 3 tensors in a vector
	return std::vector<at::Tensor>{
		std::move(nodeFeat),
		std::move(edgeFeat),
		std::move(edgeIndex)
	};
}

// This is the function exported to Python with pybind11.
// It receives a text string representing a molecule in SMILES format,
// and returns a vector contining 6 tensors, i.e. 3 for each of 2 subgraphs.
// The 3 tensors for each subgraph are: node features, edge features, edge index.
std::vector<at::Tensor>
smilesToRandomSubgraphPair(const std::string& smilesString, const double fractionNodesToRemove) {
	// Get the molecule graph data in a useful format
	GraphData graph = readGraph(smilesString);

	// Set up the graph representation in CSR (compressed sparse row) format,
	// for faster graph traversal in randomSubgraph.
	NeighbourData neighbourData = constructNeighbours(graph);

	// Allocate memory for use by randomSubgraph, reusing memory and using a single
	// allocation for multiple arrays, to reduce the number of allocations.
	// NOTE: The +1 is for consistency with the previous code that also removed
	//       one more than the floor.
	size_t numNodesToRemove = size_t(graph.numAtoms * fractionNodesToRemove) + 1;
	std::unique_ptr<uint8_t[]> visited(new uint8_t[graph.numAtoms]);
	std::unique_ptr<size_t[]> neighbourQueue(new size_t[graph.numAtoms + graph.numAtoms + 2*graph.numBonds]);
	size_t*const nodesToKeep = neighbourQueue.get() + graph.numAtoms;
	size_t*const edgesToKeep = nodesToKeep + graph.numAtoms;

	RNG& rng = getRNG();

	// Generate a random subgraph
	Subgraph subgraph0 = randomSubgraph(
		rng,
		graph.numAtoms,
		visited.get(),
		neighbourData.neighbourStarts,
		neighbourData.neighbours,
		neighbourData.bondIndices,
		neighbourQueue.get(),
		numNodesToRemove,
		nodesToKeep,
		edgesToKeep
	);

	Options& options = getOptions();
	const bool compatibleFeatureVectors = options.compatibleFeatureVectors;
	const bool useFP16 = options.useFP16;

	// Create the features for this subgraph before creating the next
	// random subgraph, because some of the data pointed to by subgraph0
	// would be overwritten in the next call to randomSubgraph.
	at::Tensor nodeFeat0;
	at::Tensor edgeFeat0;
	at::Tensor edgeIndex0;
	if (useFP16) {
		createFeatureTensors<true>(graph, compatibleFeatureVectors, &subgraph0, nodeFeat0, edgeFeat0, edgeIndex0);
	}
	else {
		createFeatureTensors<false>(graph, compatibleFeatureVectors, &subgraph0, nodeFeat0, edgeFeat0, edgeIndex0);
	}

	// Generate the next random subgraph and then its features
	Subgraph subgraph1 = randomSubgraph(
		rng,
		graph.numAtoms,
		visited.get(),
		neighbourData.neighbourStarts,
		neighbourData.neighbours,
		neighbourData.bondIndices,
		neighbourQueue.get(),
		numNodesToRemove,
		nodesToKeep,
		edgesToKeep
	);

	at::Tensor nodeFeat1;
	at::Tensor edgeFeat1;
	at::Tensor edgeIndex1;
	if (useFP16) {
		createFeatureTensors<true>(graph, compatibleFeatureVectors, &subgraph1, nodeFeat1, edgeFeat1, edgeIndex1);
	}
	else {
		createFeatureTensors<false>(graph, compatibleFeatureVectors, &subgraph1, nodeFeat1, edgeFeat1, edgeIndex1);
	}

	// Return all 6 tensors in a vector
	return std::vector<at::Tensor>{
		std::move(nodeFeat0),
		std::move(edgeFeat0),
		std::move(edgeIndex0),
		std::move(nodeFeat1),
		std::move(edgeFeat1),
		std::move(edgeIndex1)
	};
}

std::vector<size_t> randomOrder(size_t n, uint64_t seed) {
    RNG rng(seed);
    std::vector<size_t> v(n);
    std::iota(v.begin(), v.end(), 0);
    for (size_t i = 0; i < n-1; ++i) {
        const size_t source = i + (rng.random() % (n-i));
        std::swap(v[i], v[source]);
    }
    return v;
}

#if STANDALONE
// This is the main function if compiling a standalone application
// instead of a library for use by Python.
// It goes through a data set file line by line (skipping the first and loading them in 16MB chunks),
// and tests performance of smilesToRandomSubgraphPair, optionally reporting stats
// about the molecules being processed.
int main(int argc, char** argv) {
	const char*const filename = "support_data/gstore/data/resbioai/antibiotics/dataset/zinc/zinc15_cell_screening_GNE_all_081320_normalized_unique.csv";
	FILE* file = fopen(filename, "rb");
	constexpr size_t bufferSize = 16*1024*1024;
	constexpr size_t batchSize = 64*1024;
	constexpr double fractionNodesToRemove = 0.2;
	std::unique_ptr<char[]> buffer(new char[bufferSize]);
	char*const bufferBegin = buffer.get();
	std::vector<std::string> smilesStrings;
	size_t remaining = 0;
	bool isFirstLine = true;
	size_t moleculeCount = 0;
	while (true) {
		// Fill the 16MB buffer, after whatever was remaining from the last
		// line of the last buffer.
		size_t numToRead = bufferSize-remaining;
		size_t count = fread(buffer.get()+remaining, 1, numToRead, file);
#if DEBUG_LOGGING
		printf("Read %zu bytes from file\n", count);
#endif
		const bool isLast = (count < numToRead);
		count += remaining;
		const char*const currentBufferEnd = bufferBegin + count;

		const char* begin = bufferBegin;
		while (true) {
			const char* end = begin;
			while (end < currentBufferEnd && *end != '\n' && *end != '\r') {
				++end;
			}
			if (end == currentBufferEnd && !isLast) {
				// Reached the end of the 16MB buffer before the end of the line.
				// Copy the partial line at the end of this buffer to the beginning,
				// for use with the next.
				remaining = end - begin;
				memmove(bufferBegin, begin, remaining);
				break;
			}
			// We have a full line of text here.
			// First line is just "SMILES", not a molecule.
			if (!isFirstLine) {
				// Handle this line as a molecule.
				smilesStrings.push_back(std::string(begin, end));
				if (smilesStrings.size() == batchSize || isLast) {
					auto startTime = std::chrono::steady_clock::now();
					for (auto&& s : smilesStrings) {
#if DEBUG_LOGGING
						printf("Parsing molecule %zu: %s\n", moleculeCount, s.c_str());
#endif
						smilesToRandomSubgraphPair(std::string(s.c_str()), fractionNodesToRemove);
						++moleculeCount;
					}
					auto endTime = std::chrono::steady_clock::now();
					std::chrono::duration<double> elapsedTimeSeconds = endTime-startTime;
					printf("Batch of %zu molecules took %f ms (%f ms average) total %zu molecules processed\n",
						batchSize,
						(elapsedTimeSeconds.count() * 1000),
						(elapsedTimeSeconds.count() * 1000)/batchSize,
						moleculeCount
					);
#if REPORT_STATS
					size_t numUsedElements = 0;
					for (size_t i = 0; i < STATS_NUM_ELEMENTS; ++i) {
						if (statsElementCounts[i] != 0) {
							++numUsedElements;
							printf("%zu (%zu), ", i, statsElementCounts[i]);
						}
					}
					printf("\n# used elements: %zu\n", numUsedElements);
					if (moleculeCount % (16*batchSize) == 0 || isLast) {
						printf("\n");
						printf("Atom counts per molecule:\n");
						for (size_t i = 0; i < STATS_NUM_MOL_ATOM_COUNTS; ++i) {
							printf("%zu\t%zu\t%zu\n", i, statsMolAtomCounts[i], statsMolBondCounts[i]);
						}
						printf("Degree histogram:");
						for (size_t i = 0; i < STATS_NUM_DEGREES; ++i) {
							printf(" %zu", statsDegreeCounts[i]);
						}
						printf("\n");
						printf("Charge histogram:");
						for (size_t i = 0; i < STATS_NUM_CHARGES; ++i) {
							printf(" %zu", statsChargeCounts[i]);
						}
						printf("\n");
						printf("Chirality histogram:");
						for (size_t i = 0; i < STATS_NUM_CHIRALITIES; ++i) {
							printf(" %zu", statsChiralityCounts[i]);
						}
						printf("\n");
						printf("Hs histogram:");
						for (size_t i = 0; i < STATS_NUM_HS; ++i) {
							printf(" %zu", statsHCounts[i]);
						}
						printf("\n");
						printf("Hybridization histogram:");
						for (size_t i = 0; i < STATS_NUM_HYBRIDIZATIONS; ++i) {
							printf(" %zu", statsHybridizationCounts[i]);
						}
						printf("\n");
						printf("Aromatic atoms: %zu/%zu\n", statsAromaticAtomCount, statsTotalNumAtoms);
						printf("\n");
						printf("Bond type histogram:");
						for (size_t i = 0; i < STATS_NUM_BOND_TYPES; ++i) {
							printf(" %zu", statsBondTypeCounts[i]);
						}
						printf("\n");
						printf("Bond stereo histogram:");
						for (size_t i = 0; i < STATS_NUM_BOND_STEREOS; ++i) {
							printf(" %zu", statsBondStereoCounts[i]);
						}
						printf("\n");
						printf("Conjugated bonds: %zu/%zu\n", statsConjugatedBondCount, statsTotalNumBonds);
						printf("Bonds in rings: %zu/%zu\n", statsBondInRingCount, statsTotalNumBonds);
						printf("\n");
					}
#endif
					fflush(stdout);
					smilesStrings.resize(0);
				}
			}
			isFirstLine = false;
			begin = end;
			while (begin < currentBufferEnd && (*begin == '\n' || *begin == '\r')) {
				++begin;
			}
			if (begin == currentBufferEnd && isLast) {
				break;
			}
		}
		if (begin == currentBufferEnd && isLast) {
			break;
		}
	}
	fclose(file);
	return 0;
}
#else
// This is necessary to export smilesToRandomSubgraphPair as a Python function
// with the same name, in a Python module named GNEpropCPP.
PYBIND11_MODULE(GNEpropCPP, m) {
	m.doc() = "GNEprop C++ plugin"; // Python module docstring
	m.def("smilesToGraph", &smilesToGraph, "Accepts a SMILES string and returns a graph");
	m.def("smilesToRandomSubgraph", &smilesToRandomSubgraph, "Accepts a SMILES string and returns 1 random subgraph");
	m.def("smilesToRandomSubgraphPair", &smilesToRandomSubgraphPair, "Accepts a SMILES string and returns 2 random subgraphs");
	m.def("randomOrder", &randomOrder, "Returns a list of integers from 0 to n-1 in a random order");
	m.def("configCPPOptions", &configCPPOptions, "Sets GNEpropCPP options");
}
#endif
