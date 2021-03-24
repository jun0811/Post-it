package com.ssafy.service;

import com.ssafy.entity.YoutubeDto;
import com.ssafy.repository.YoutubeRepository;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class YoutubeServiceImpl implements YoutubeService {

    private final YoutubeRepository youtubeRepository;

    public YoutubeServiceImpl(YoutubeRepository youtubeRepository) {
        this.youtubeRepository = youtubeRepository;
    }

    @Override
    public List<YoutubeDto> listYoutubeContents() {
        /*
        1. findAll함수로 List<YoutubeDto>를 가져옴.
        2. Optinal로 null체크
            1. null -> 빈List만들어서 리턴
            2. 아닐경우 스트림으로 변경 -> 최신순으로 정렬 -> list형태로 변환 후 리턴
        */
        return Optional.ofNullable(youtubeRepository.findAll()).orElseGet(Collections::emptyList)
                .stream()
                .sorted(((o1, o2) -> o2.getDate().compareTo(o1.getDate())))
                .collect(Collectors.toList());
    }
}