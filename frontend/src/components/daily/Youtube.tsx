import React, { useState, useEffect, useLayoutEffect } from 'react';
import styled from 'styled-components';
import Grid from '@material-ui/core/Grid';
import { TurnedIn } from '@material-ui/icons';
import { SliderSwitch, StyledCard } from './Daily.styles';
import { allYoutube } from 'api/daily';
import LazyLoad from 'react-lazyload';
import { CardButtonGroup, Switch } from './Common';

// Base title
const Title = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.xl};
  color: ${({ theme }) => theme.colors.text.first};
  display: flex;
  align-items: center;
`;

const SubTitle = styled.div`
  font-size: ${({ theme }) => theme.fontSizes.lg};
  color: ${({ theme }) => theme.colors.card.content};
  margin: 10px;
  display: flex;
  justify-content: space-between;
`;

const CardButtonWrapper = styled.div`
  display: flex;
  /* align-items: center; */
`;

function Youtube() {
  const [youtube, setYoutube] = useState([] as any);
  const [tmp, setTmp] = useState([] as any);

  const [youtubeId, setYoutubeId] = useState([] as any);

  useEffect(() => {
    async function setContent() {
      const data = await allYoutube();
      setYoutube(data.data.data);
      setTmp(data.data.data);
    }
    setContent();
    console.log(youtubeId);

    return () => {};
  }, []);

  function idAdd(data: any) {
    setYoutubeId(youtubeId.concat(data));
  }

  function idRemove(data: any) {
    setYoutubeId(youtubeId.filter((id: any) => data != id));
  }

  const cardList = youtube.map((res: any) => (
    <Grid item xs={12} md={4} sm={6}>
      <StyledCard
        style={{
          borderRadius: '20px',
          height: '450px',
          backgroundColor: '#2e2e2e',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <img
          src={`https://img.youtube.com/vi/${res.youtubeId}/maxresdefault.jpg`}
          alt="content image"
          style={{ objectFit: 'fill' }}
        />
        <div className="content">
          <div className="inner">
            <SubTitle>
              <a href={res.url}>{res.title}</a>
              <CardButtonGroup
                checked={youtubeId}
                id={res.id}
                idAdd={idAdd}
                idRemove={idRemove}
              ></CardButtonGroup>
            </SubTitle>
            <SubTitle style={{ backgroundColor: '#201d29', marginTop: 'auto' }}>
              <p></p>
              <p>{res.date}</p>
            </SubTitle>
          </div>
        </div>
      </StyledCard>
    </Grid>
  ));

  // 토글 스위치 함수
  function filterCard(data: boolean) {
    console.log(data);
    if (data == true) {
      setYoutube(
        youtube.filter((res: any) => youtubeId.includes(res.id)) as any,
      );
    } else {
      setYoutube(tmp);
    }
  }

  return (
    <div>
      <Title> 유튜브 동영상</Title>
      <Title style={{ fontSize: '16px', float: 'right' }}>
        즐겨찾기 <Switch filterCard={filterCard}></Switch>
      </Title>
      <br />
      <Grid spacing={4}>
        <Grid item xs={12}>
          <Grid container spacing={4}>
            {cardList}
          </Grid>
        </Grid>
      </Grid>
    </div>
  );
}

export default Youtube;
